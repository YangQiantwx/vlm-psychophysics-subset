"""Plan, run and verify benchmark evaluations on model APIs."""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
import csv
import fcntl
import hashlib
from http.client import HTTPException
from importlib.metadata import version
from io import StringIO
import json
import os
from pathlib import Path
import re
import sqlite3
from tempfile import TemporaryDirectory
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

from .. import catalog as release
from ..release import select_samples
from . import native_video_api
from .provider_api import ROUTES

SCHEMA = "provider-eval-v1"
LEGACY_SCHEMA_SHA256 = "1b616bcee9cf4494f3e0befe71d6404fe1e643ea1b5573f9c3d6f32c5978bec6"
PROVIDER = "nvidia-internal"
BASE_URL, KEY_ENV, _ = ROUTES[PROVIDER]
NVIDIA_REPEAT_PRESET = "nvidia-repeat"
NVIDIA_REPEAT_GENERATION = {"n": 1, "temperature": 0.0, "top_p": 1.0, "seed": 0}
GATEWAY_VIDEO_DELIVERIES = ("native", "ordered-images")
GATEWAY_IMAGE_DETAILS = ("auto", "low", "high", "original")
GATEWAY_TOKEN_FIELDS = ("max_completion_tokens", "max_tokens")
ORDERED_IMAGE_VIDEO_PROMPT_PREFIX = (
    "The video is provided as 180 consecutive frames in chronological order at 60 fps.\n\n"
)
TASK_SCOPES = {
    f"{task}-{domain}": scope
    for scope in release.SCOPES
    for domain, task in [scope.split("/")]
}
RETRIES = {"SOURCE_ERROR", "CLAIMED", "HTTP_ERROR"}
UNCERTAIN = {"SENT", "TRANSPORT_UNKNOWN"}
LEGACY_RETURN_FILES = (
    "plan.json",
    "source-plan.jsonl",
    "config.json",
    "attempts.jsonl",
    "results.jsonl",
    "summary.json",
)
RETURN_FILES = (*LEGACY_RETURN_FILES, "results.csv")
CSV_FIELDS = (
    "sample_id",
    "scope",
    "input",
    "requested_model",
    "returned_model",
    "truth",
    "raw_answer",
    "normalized_answer",
    "accuracy_eligible",
    "correct",
    "strict_format_valid",
    "status",
    "request_id",
    "source_video_sha256",
    "prompt_sha256",
    "static_case_id",
)


def known_schema(value):
    """Read the original return format as well as the current neutral identifier."""
    return isinstance(value, str) and (
        value == SCHEMA or hashlib.sha256(value.encode()).hexdigest() == LEGACY_SCHEMA_SHA256
    )


def atomic(path, data):
    temp = path.with_name("." + path.name + ".partial")
    with temp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_rows(raw):
    return [json.loads(line) for line in raw.splitlines()]


def load_plan(out):
    plan = json.loads((out / "plan.json").read_bytes())
    release.require(
        known_schema(plan["schema"])
        and release.known_public_release(plan["revision"], plan["index_sha256"]),
        "Unsupported run plan",
    )
    raw = (out / "source-plan.jsonl").read_bytes()
    rows = load_rows(release.checked(raw, plan["source_plan_sha256"], "source plan"))
    release.require(release.counts(rows) == plan["profiles"], "Plan counts differ")
    return plan, rows


def build_plan(
    out, metadata_root=None, *, scopes=None, modality=None,
    samples=None, selectors=None, index_bounds=None, download_missing=False,
):
    if download_missing:
        rows = release.public_plan(metadata_root, scopes, modality, download_missing=True)
    elif scopes or modality:
        rows = release.public_plan(metadata_root, scopes, modality)
    else:
        rows = release.public_plan(metadata_root)
    rows = filter_samples(rows, samples, selectors)
    rows.sort(key=lambda r: (r["scope"], r["input"], r["sample_id"]))
    selection = None
    if index_bounds is not None:
        release.require(
            not samples and not selectors and scopes and len(scopes) == 1 and modality,
            "Index selection requires one public task and modality",
        )
        selection = index_selection(index_bounds, scopes[0], modality, len(rows))
        rows = rows[selection["from_index"]:selection["to_index"]]
    raw = b"".join(map(release.encoded, rows))
    plan = dict(
        schema=SCHEMA,
        repo=release.REPO,
        revision=release.REVISION,
        index_sha256=release.INDEX_SHA,
        private_index_sha256=None,
        source_plan_sha256=release.sha(raw),
        profiles=release.counts(rows),
        logical_rows=len(rows),
        unique_requests=len(release.unique_requests(rows)),
        static_policy="one provider call per frozen static_case_id; project to matching video IDs",
        t5_static_policy="reference first, then event midpoints in chronological order",
    )
    if selection is not None:
        plan["index_selection"] = selection
    if (out / "plan.json").exists():
        release.require(
            load_plan(out)[0] == plan, "This output directory already has another plan"
        )
    else:
        atomic(out / "source-plan.jsonl", raw)
        atomic(out / "plan.json", release.encoded(plan))
    return plan, rows


def requested_frame_settings(config):
    if config.get("video_delivery") == "ordered-images":
        return native_video_api.ordered_image_settings(PROVIDER)
    settings = native_video_api.frame_settings(config["provider"], config.get("video_fps"))
    if config.get("preset") == NVIDIA_REPEAT_PRESET:
        return {
            **settings,
            "mm_processor_kwargs": {**settings["mm_processor_kwargs"], "do_resize": False},
        }
    return settings


def configured(
    out, plan, model, max_tokens, provider=PROVIDER, all_frames=False,
    generation_parameters=None, preset=None,
    *, video_delivery=None, image_detail=None, token_field=None,
):
    release.require(
        bool(model) and bool(re.fullmatch(r"[A-Za-z0-9_.:/-]+", model)),
        "Set --model to the exact provider model ID",
    )
    hosted = native_video_api.HOSTED.get(provider)
    endpoint = BASE_URL
    if hosted:
        endpoint = native_video_api.endpoint(provider)
    elif provider == "vllm":
        endpoint = native_video_api.vllm_endpoint(os.environ.get("VLLM_BASE_URL", ""))
    config = dict(
        schema=SCHEMA,
        provider=provider,
        endpoint=endpoint,
        requested_model=model,
        max_completion_tokens=max_tokens,
        source_plan_sha256=plan["source_plan_sha256"],
        dataset_revision=plan["revision"],
        dataset_index_sha256=plan["index_sha256"],
        code_sha256={
            p.name: file_sha(p)
            for p in (
                Path(__file__),
                Path(release.__file__),
                Path(__file__).with_name("provider_api.py"),
            )
        },
        generation="provider defaults apart from token limit; no tools; no schema constraint",
        transport="inline MIME-correct data URLs, image/reference then video as applicable",
        processor_visibility="not exposed by provider",
        answer_parser="single-label-v1",
    )
    if hosted:
        config.update(
            video_fps=native_video_api.requested_fps(provider, model),
            transport="inline video bytes",
        )
        config["code_sha256"]["native_video_api.py"] = file_sha(
            Path(native_video_api.__file__)
        )
        if provider == "claude":
            config["api_version"] = native_video_api.CLAUDE_API_VERSION
        native_video_api.validate_config(config)
        if provider in native_video_api.FRAME_LISTS:
            config["transport"] = "ordered lossless PNG images"
            config["media_runtime"] = {name: version(name) for name in ("av", "Pillow")}
        if provider == "astra":
            generation_parameters = {
                "reasoning": {"effort": "medium"},
                **(generation_parameters or {}),
            }
    if generation_parameters:
        config["generation_parameters"] = generation_parameters
        config["generation"] = "explicit requested settings, no tools"
    if preset == NVIDIA_REPEAT_PRESET:
        config["preset"] = preset
        config["processor_visibility"] = "provider controls frame sampling and geometry"
    if provider == PROVIDER and preset is None:
        config["video_delivery"] = video_delivery or "native"
        if image_detail is not None:
            config["image_detail"] = image_detail
        config["token_field"] = token_field or "max_completion_tokens"
        if not all_frames:
            config["pilot_admission"] = "completion_envelope"
        if config["video_delivery"] == "ordered-images":
            config.update(
                transport="ordered lossless PNG images in Chat Completions",
                processor_visibility="client verifies the submitted frames; server processor is unobserved",
                video_prompt_prefix=ORDERED_IMAGE_VIDEO_PROMPT_PREFIX,
                video_delivery_settings=native_video_api.ordered_image_settings(PROVIDER),
                media_runtime={name: version(name) for name in ("av", "Pillow")},
            )
            config["code_sha256"]["native_video_api.py"] = file_sha(
                Path(native_video_api.__file__)
            )
    if all_frames:
        config["frame_settings"] = requested_frame_settings(config)
        config["processor_visibility"] = "requires separate server frame report"
        if provider in native_video_api.ORDERED_IMAGES:
            config["video_prompt_prefix"] = ORDERED_IMAGE_VIDEO_PROMPT_PREFIX
        config["code_sha256"]["native_video_api.py"] = file_sha(
            Path(native_video_api.__file__)
        )
    path = out / "config.json"
    if path.exists():
        release.require(
            json.loads(path.read_bytes()) == config,
            "Model, request parameters, release, or runner changed. Use a new output directory.",
        )
    else:
        atomic(path, release.encoded(config))
    return config


def open_db(out):
    db = sqlite3.connect(out / "attempts.sqlite3", timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute(
        """CREATE TABLE IF NOT EXISTS attempts (
        request_id TEXT NOT NULL, attempt INTEGER NOT NULL, details BLOB NOT NULL, body BLOB,
        PRIMARY KEY (request_id, attempt))"""
    )
    return db


def save(db, receipt, body=None, new=False):
    if new:
        db.execute(
            "INSERT INTO attempts VALUES (?, ?, ?, ?)",
            (receipt["request_id"], receipt["attempt"], release.encoded(receipt), body),
        )
    else:
        changed = db.execute(
            "UPDATE attempts SET details=?, body=? WHERE request_id=? AND attempt=?",
            (release.encoded(receipt), body, receipt["request_id"], receipt["attempt"]),
        )
        release.require(changed.rowcount == 1, "Attempt disappeared")
    db.commit()


def records(db):
    rows = defaultdict(list)
    for (data,) in db.execute(
        "SELECT details FROM attempts ORDER BY request_id, attempt"
    ):
        row = json.loads(data)
        rows[row["request_id"]].append(row)
    return rows


def accepted_pilot(attempt, config):
    return attempt["phase"] == "pilot" and (
        attempt.get("modality_admitted")
        or (
            config.get("pilot_admission") == "completion_envelope"
            and attempt["status"] in {"FORMAT_OR_TOOL_FAILURE", "TRUNCATED"}
            and attempt.get("returned_model")
        )
    )


def validate_frame_report(report, attempt, config):
    indices = list(range(native_video_api.SOURCE_FRAMES))
    release.require(
        attempt["phase"] == "pilot"
        and attempt.get("modality_admitted")
        and report["endpoint"] == config["endpoint"]
        and report["model"] == attempt["returned_model"]
        and report["request_sha256"] == attempt["request_sha256"]
        and isinstance(report["deployment"], str)
        and bool(report["deployment"].strip())
        and report["decoded_indices"] == report["processor_indices"] == indices
        and (
            "dataset_revision" not in config
            or all(
                type(i) is int
                for key in ("decoded_indices", "processor_indices")
                for i in report[key]
            )
        ),
        "The server report must bind the successful pilot and show decoded and processor indices 0–179 in order",
    )
    for key in ("resize_policy", "crop_policy"):
        if key in report:
            release.require(
                isinstance(report[key], str) and bool(report[key].strip()),
                f"{key} must describe the effective server policy",
            )
    if "bit_depth" in report:
        release.require(
            type(report["bit_depth"]) is int and report["bit_depth"] > 0,
            "bit_depth must be a positive integer",
        )

    if config.get("preset") == NVIDIA_REPEAT_PRESET:
        release.require(
            (report.get("width"), report.get("height"), report.get("fps"))
            == (
                native_video_api.SOURCE_WIDTH, native_video_api.SOURCE_HEIGHT,
                native_video_api.SOURCE_FPS,
            )
            and report.get("resize_policy") == report.get("crop_policy") == "none"
            and report.get("bit_depth") == 8,
            "The NVIDIA all-frame report must show 768×512 RGB8 at 60 fps with no resizing or cropping",
        )
        if "generation_settings" in report:
            release.require(
                report["generation_settings"] == config["generation_parameters"],
                "The server's generation settings differ from the repeat preset",
            )


def attach_frame_reports(path, db, history, config):
    reports = json.loads(path.read_bytes())
    release.require(
        isinstance(reports, list) and bool(reports),
        "Expected a nonempty list of server reports",
    )
    pilots = {
        a["request_sha256"]: a
        for attempts in history.values()
        for a in attempts
        if a["phase"] == "pilot" and a.get("modality_admitted")
    }
    staged = {}
    deployments = {
        a["frame_report"]["deployment"] for a in pilots.values() if "frame_report" in a
    }
    for report in reports:
        release.require(isinstance(report, dict), "Expected a server report object")
        attempt = pilots.get(report["request_sha256"])
        release.require(
            attempt is not None,
            "Server report does not match a successful pilot in this run",
        )
        validate_frame_report(report, attempt, config)
        evidence = {
            key: report[key]
            for key in (
                "endpoint",
                "model",
                "deployment",
                "request_sha256",
                "decoded_indices",
                "processor_indices",
            )
        }
        evidence.update({
            key: report[key]
            for key in (
                "resize_policy", "crop_policy", "bit_depth",
                "width", "height", "fps", "generation_settings",
            )
            if key in report
        })
        digest = attempt["request_sha256"]
        prior = staged.get(digest, attempt.get("frame_report", {}))
        release.require(
            all(prior[key] == evidence[key] for key in prior.keys() & evidence.keys()),
            "Server frame evidence differs from the accepted report",
        )
        staged[digest] = {**prior, **evidence}
        deployments.add(evidence["deployment"])
    release.require(
        len(deployments) == 1,
        "Use a separate output directory for each server deployment",
    )
    with db:
        for digest, report in staged.items():
            attempt = pilots[digest]
            attempt["frame_report"] = report
            db.execute(
                "UPDATE attempts SET details=? WHERE request_id=? AND attempt=?",
                (release.encoded(attempt), attempt["request_id"], attempt["attempt"]),
            )


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def transport(payload, credential, config=None):
    endpoint = config["endpoint"] if config else BASE_URL
    timeout = 300
    if config and config["provider"] == "gemini":
        endpoint += "/models/" + config["requested_model"] + ":generateContent"
        auth = {"x-goog-api-key": credential}
    elif config and config["provider"] == "astra":
        endpoint += "/responses"
        timeout = 900
        auth = {"Authorization": "Bearer " + credential}
        if project := os.environ.get("OPENAI_PROJECT_ID"):
            auth["OpenAI-Project"] = project
    elif config and config["provider"] == "claude":
        endpoint += "/messages"
        timeout = 600
        auth = {
            "Authorization": "Bearer " + credential,
            "anthropic-version": config["api_version"],
        }
        if workspace := os.environ.get("ANTHROPIC_WORKSPACE_ID"):
            auth["anthropic-workspace-id"] = workspace
    else:
        endpoint += "/chat/completions"
        auth = {"Authorization": "Bearer " + credential}
    request = Request(
        endpoint,
        data=payload,
        headers={**auth, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = build_opener(NoRedirect).open(request, timeout=timeout)
    except HTTPError as response:
        with response:
            return response.code, response.headers, response.read()
    with response:
        return response.status, response.headers, response.read()


def response_fields(body, status, choices, expected_model, provider=PROVIDER):
    if not 200 <= status < 300:
        return {"status": "HTTP_ERROR"}
    try:
        document = json.loads(body)
        if provider == "gemini":
            document = native_video_api.gemini_completion(document)
        elif provider == "astra":
            document = native_video_api.astra_completion(document)
        elif provider == "claude":
            document = native_video_api.claude_completion(document)
        options = document["choices"]
        if len(options) != 1:
            raise ValueError("Expected one completion choice")
        option = options[0]
        message = option["message"]
        raw = message.get("content")
        model = document["model"]
        finish = option.get("finish_reason")
        tools = bool(message.get("tool_calls") or message.get("function_call"))
        if not isinstance(model, str) or not model:
            raise ValueError("Missing returned model identity")
    except (KeyError, IndexError, TypeError, AttributeError, ValueError):
        return {"status": "RESPONSE_ENVELOPE_ERROR"}
    complete = (
        isinstance(raw, str) and bool(raw.strip()) and finish == "stop" and not tools
    )
    label = re.sub(r"[.!]$", "", raw.strip().upper()) if complete else None
    if label not in choices:
        label = None
    status_name = (
        "MODEL_ID_CHANGED"
        if expected_model and model != expected_model
        else (
            "TRUNCATED"
            if finish == "length"
            else "ANSWERED" if label is not None else "FORMAT_OR_TOOL_FAILURE"
        )
    )
    return dict(
        status=status_name,
        returned_model=model,
        response_id=document.get("id"),
        system_fingerprint=document.get("system_fingerprint"),
        usage=document.get("usage"),
        finish_reason=finish,
        tool_calls_returned=tools,
        raw_answer=raw,
        normalized_answer=label if status_name == "ANSWERED" else None,
        strict_format_valid=complete and raw in choices and status_name == "ANSWERED",
        modality_admitted=status_name == "ANSWERED",
    )


def request_prompt(row, config):
    if row["input"] == "video":
        return config.get("video_prompt_prefix", "") + row["prompt"]
    return row["prompt"]


def request_body(row, config, media_root, *, download_missing=False):
    release.require(row["source"] == "public", "Select a published dataset trial.")
    provider = config.get("provider", PROVIDER)
    gateway_frames = config.get("video_delivery") == "ordered-images"
    media = []
    frame_parts = []
    verified = []
    for item in row["media"]:
        local = media_root / item["path"] if media_root is not None else None
        if local is None:
            data = release.public_media(item)
        else:
            if download_missing and not local.is_file():
                local = release.public_file(item["path"], media_root)
            data = local.read_bytes()
        release.checked(data, item["sha256"], item["path"])
        verified.append({**item, "bytes": len(data)})
        if gateway_frames or provider in native_video_api.FRAME_LISTS:
            if item["mime"] == "image/png":
                url = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
                frame_parts.append({"type": "image_url", "image_url": {"url": url}})
            else:
                release.require(item["mime"] == "video/mp4", "Use the verified MP4 release")
                images, rgb_sha = native_video_api.ordered_pngs(
                    data,
                    item.get("processor_input_rgb_sha256"),
                    compression=native_video_api.PNG_COMPRESSION[provider],
                )
                part = {"type": "video", "video": images}
                if provider == "qwen-frames":
                    pixels = native_video_api.SOURCE_WIDTH * native_video_api.SOURCE_HEIGHT
                    part.update(
                        min_pixels=pixels,
                        max_pixels=pixels,
                        total_pixels=len(images) * pixels,
                    )
                    if config["video_fps"] is not None:
                        part["fps"] = config["video_fps"]
                frame_parts.append(part)
                verified[-1]["ordered_frames"] = {
                    "count": len(images),
                    "rgb_sha256": rgb_sha,
                    "fps": native_video_api.SOURCE_FPS,
                }
        else:
            media.append((item["mime"], base64.b64encode(data).decode("ascii")))
    generation = config.get("generation_parameters", {})
    if provider in native_video_api.FRAME_LISTS:
        body = native_video_api.frame_list_request(
            provider,
            config["requested_model"],
            request_prompt(row, config),
            frame_parts,
            config["max_completion_tokens"],
            generation,
        )
    elif provider in native_video_api.HOSTED:
        body = native_video_api.request(
            provider,
            config["requested_model"],
            row["prompt"],
            media,
            config["max_completion_tokens"],
            config["video_fps"],
        )
    else:
        parts = []
        for mime, data in media:
            release.require(mime in {"image/png", "video/mp4", "video/x-matroska"}, "Unsupported media MIME type")
            kind = "image_url" if mime == "image/png" else "video_url"
            parts.append({"type": kind, kind: {"url": f"data:{mime};base64,{data}"}})
        for part in frame_parts:
            if part["type"] == "video":
                parts.extend({"type": "image_url", "image_url": {"url": url}} for url in part["video"])
            else:
                parts.append(part)
        if detail := config.get("image_detail"):
            for part in parts:
                if part["type"] == "image_url":
                    part["image_url"]["detail"] = detail
        body = {
            "model": config["requested_model"],
            "messages": [{"role": "user", "content": [
                *parts, {"type": "text", "text": request_prompt(row, config)},
            ]}],
            "stream": False,
            config.get("token_field", "max_completion_tokens"): config["max_completion_tokens"],
        }
        if not gateway_frames:
            body.update(config.get("frame_settings", {}))
    if provider not in native_video_api.FRAME_LISTS:
        body.update(generation)
    payload = release.encoded(body)
    native_video_api.validate_payload_size(provider, config.get("endpoint"), len(payload))
    return payload, verified


def invoke(out, receipt, row, config, media_root, credential, expected_model, download_missing):
    with closing(open_db(out)) as db:
        started = time.monotonic()
        try:
            payload, verified = request_body(
                row, config, media_root, download_missing=download_missing
            )
        except Exception as error:
            # Exception messages from network clients can contain signed URLs.
            receipt.update(
                status="SOURCE_ERROR",
                error_type=type(error).__name__,
                source_error=str(error) if isinstance(error, ValueError) else None,
            )
            receipt.update(
                finished_unix=time.time(), elapsed_seconds=time.monotonic() - started
            )
            save(db, receipt)
            return receipt
        receipt.update(
            status="SENT",
            request_sha256=release.sha(payload),
            request_bytes=len(payload),
            verified_media=verified,
            sent_unix=time.time(),
        )
        if "video_prompt_prefix" in config or "video_delivery" in config or config.get("preset") == NVIDIA_REPEAT_PRESET:
            receipt["request_prompt_sha256"] = release.sha(
                request_prompt(row, config).encode("utf-8")
            )
        save(db, receipt)
        raw = None
        try:
            http_status, headers, raw = transport(payload, credential, config)
            receipt.update(
                http_status=http_status,
                response_body_sha256=release.sha(raw),
                response_bytes=len(raw),
            )
            if credential.encode() in raw:
                receipt.update(
                    status="SENSITIVE_RESPONSE_WITHHELD",
                    response_body_withheld="credential_echo",
                )
                raw = None
            elif re.search(
                rb"data:(?:image|video)/[a-z0-9.-]+;base64,", raw, re.IGNORECASE
            ) or (
                config["provider"] == "gemini"
                and re.search(rb'"inline_?data"\s*:', raw, re.IGNORECASE)
            ) or (
                config["provider"] == "claude"
                and re.search(rb'\\?"type\\?"\s*:\s*\\?"base64\\?"', raw, re.IGNORECASE)
            ):
                receipt.update(
                    status="SENSITIVE_RESPONSE_WITHHELD",
                    response_body_withheld="media_echo",
                )
            else:
                receipt.update(
                    provider_request_id=headers.get("x-request-id")
                    or headers.get("request-id"),
                    retry_after=headers.get("retry-after"),
                    **response_fields(
                        raw,
                        http_status,
                        row["choices"],
                        expected_model,
                        config["provider"],
                    ),
                )
        except (OSError, HTTPException) as error:
            receipt.update(status="TRANSPORT_UNKNOWN", error_type=type(error).__name__)
        receipt.update(
            finished_unix=time.time(), elapsed_seconds=time.monotonic() - started
        )
        save(db, receipt, raw)
        return receipt


def filter_samples(rows, samples=None, selectors=None):
    if selectors:
        selected = select_samples(
            ((row["scope"], row["sample_id"], row["source_video"]["path"]) for row in rows),
            selectors,
        )
        return [row for row in rows if (row["scope"], row["sample_id"]) in selected]
    if not samples:
        return rows
    wanted = set(samples)
    found = set()
    selected = []
    for row in rows:
        matches = wanted.intersection((row["sample_id"], row["static_case_id"]))
        if matches:
            found.update(matches)
            selected.append(row)
    release.require(wanted == found, f"Unknown sample IDs: {', '.join(sorted(wanted - found))}")
    return selected


def index_selection(bounds, scope, modality, total):
    first, stop = bounds
    first = 0 if first is None else first
    stop = total if stop is None else stop
    release.require(
        0 <= first < stop <= total,
        f"Expected 0 <= from-index < to-index <= {total}, got [{first}, {stop})",
    )
    return dict(
        scope=scope, input=modality, from_index=first, to_index=stop, total_trials=total,
    )


def selected_requests(rows, args):
    matching = [
        row for row in rows
        if (not args.scope or row["scope"] in args.scope)
        and (not args.input or row["input"] == args.input)
    ]
    requests = release.unique_requests(matching)
    samples = getattr(args, "sample_id", None)
    selectors = getattr(args, "sample", None)
    if not samples and not selectors:
        return requests
    requested = {row["request_id"] for row in filter_samples(matching, samples, selectors)}
    return {key: row for key, row in requests.items() if key in requested}


def execution(out, plan, rows, args, pilot):
    release.require(
        (plan["revision"], plan["index_sha256"])
        == (release.REVISION, release.INDEX_SHA),
        "Resume an earlier release with the checkout that created the run plan.",
    )
    selected = selected_requests(rows, args)
    release.require(bool(selected), "The selection contains no callable trials")
    release.require(
        all(row["source"] == "public" for row in selected.values()),
        "Select published dataset trials for a new evaluation.",
    )
    release.require(
        args.execute and args.max_requests > 0,
        "Set --execute and a positive --max-requests",
    )
    hosted = native_video_api.HOSTED.get(args.provider)
    gateway_frames = getattr(args, "video_delivery", None) == "ordered-images"
    if args.all_frames:
        release.require(
            not hosted or args.provider in native_video_api.FRAME_LISTS,
            "This provider samples video. Choose a full-frame deployment or pass --sampled-video.",
        )
        release.require(
            all(
                r["input"] == "video"
                and (
                    r["source_video_format"]["frames"],
                    r["source_video_format"]["fps"],
                )
                == (180, 60)
                for r in selected.values()
            ),
            "Select a 180-frame, 60 fps video task for the all-frame run",
        )
    if args.all_frames and getattr(args, "preset", None) == NVIDIA_REPEAT_PRESET:
        release.require(
            all(
                row["source"] == "public"
                and (row["source_video_format"]["width"], row["source_video_format"]["height"])
                == (native_video_api.SOURCE_WIDTH, native_video_api.SOURCE_HEIGHT)
                and any(item["mime"] == "video/mp4" for item in row["media"])
                for row in selected.values()
            ),
            "The NVIDIA all-frame pilot requires a 768×512 video from the public MP4 release",
        )
    if args.provider in native_video_api.FRAME_LISTS or gateway_frames:
        release.require(
            all(
                r["source"] == "public"
                and all(i["mime"] in {"image/png", "video/mp4"} for i in r["media"])
                and (r["input"] == "static" or args.all_frames or gateway_frames)
                for r in selected.values()
            ),
            "Select public MP4 and PNG inputs for the ordered image route",
        )
    elif hosted:
        release.require(
            all(
                r["source"] == "public"
                and r["input"] == "video"
                and len(r["media"]) == 1
                and r["media"][0]["mime"] in hosted[2]
                for r in selected.values()
            ),
            "Select supported public single-video scopes with --scope and --input video",
        )
    config = configured(
        out, plan, args.model, args.max_tokens, args.provider, args.all_frames,
        getattr(args, "generation_parameters", None), preset=getattr(args, "preset", None),
        video_delivery=getattr(args, "video_delivery", None),
        image_detail=getattr(args, "image_detail", None),
        token_field=getattr(args, "token_field", None),
    )
    with closing(open_db(out)) as db:
        history = records(db)
        if args.frame_report and not pilot:
            release.require(args.all_frames, "--frame-report requires an all-frame run")
            attach_frame_reports(args.frame_report, db, history, config)
        profiles = defaultdict(list)
        for key, row in selected.items():
            profiles[row["profile"]].append(key)
        admitted = {
            r["profile"]: r["returned_model"]
            for attempts in history.values()
            for r in attempts
            if accepted_pilot(r, config)
            and (pilot or not args.all_frames or r.get("frame_report"))
        }
        models = set(admitted.values())
        release.require(len(models) <= 1, "Pilots returned different model identities")
        expected_model = next(iter(models), None)
        blocked = sorted(profiles.keys() - admitted.keys())
        if pilot:
            overrides = {}
            for key in args.pilot_trial or []:
                release.require(
                    key in selected and key not in history,
                    f"Pilot trial {key} is unknown or already asked",
                )
                profile = selected[key]["profile"]
                release.require(
                    profile in blocked and profile not in overrides,
                    f"Pilot profile {profile} is already admitted or has two overrides",
                )
                overrides[profile] = key
            keys = [
                overrides.get(profile, min(profiles[profile])) for profile in blocked
            ]
        else:
            keys = [key for key, row in selected.items() if row["profile"] in admitted]
        retry = set(args.retry or [])
        release.require(
            retry <= selected.keys(), "A retry ID is outside the callable selection"
        )
        eligible = []
        for key in keys:
            attempts = history.get(key, [])
            if not attempts:
                eligible.append(key)
            elif key in retry:
                previous = attempts[-1]
                can_retry = previous["status"] in RETRIES or (
                    args.retry_uncertain and previous["status"] in UNCERTAIN
                )
                release.require(
                    len(attempts) == 1 and can_retry,
                    f"Trial {key} cannot be retried under the current flags",
                )
                eligible.append(key)
        release.require(
            retry <= set(eligible),
            "A requested retry is outside this pilot or admitted run",
        )
        if not eligible:
            return dict(status="NO_NEW_REQUESTS", started=0, blocked_profiles=blocked)
        key_env = (
            hosted[0]
            if hosted
            else "VLLM_API_KEY" if args.provider == "vllm" else KEY_ENV
        )
        credential = os.environ.get(key_env)
        release.require(bool(credential), f"Set {key_env} in the operator shell")
        queue = iter(eligible[: args.max_requests])
        started, recent_failures = 0, 0
        stop = None
        statuses = Counter()
        workers = 1 if pilot else args.workers
        with ThreadPoolExecutor(max_workers=workers) as pool:
            running = set()

            def schedule():
                nonlocal started
                key = next(queue, None)
                if key is None:
                    return
                row = selected[key]
                receipt = dict(
                    request_id=key,
                    attempt=len(history.get(key, [])) + 1,
                    profile=row["profile"],
                    sample_id=row["sample_id"],
                    source=row["source"],
                    status="CLAIMED",
                    phase="pilot" if pilot else "run",
                    started_unix=time.time(),
                )
                save(db, receipt, new=True)
                running.add(
                    pool.submit(
                        invoke,
                        out,
                        receipt,
                        row,
                        config,
                        args.media_root,
                        credential,
                        expected_model,
                        getattr(args, "download_media", False),
                    )
                )
                started += 1

            for _ in range(workers):
                schedule()
            while running:
                finished, running = wait(running, return_when=FIRST_COMPLETED)
                for job in finished:
                    item = job.result()
                    status = item["status"]
                    statuses[status] += 1
                    if pilot and accepted_pilot(item, config):
                        admitted[item["profile"]] = item["returned_model"]
                        expected_model = item["returned_model"]
                    failed = status in {
                        "SOURCE_ERROR",
                        "HTTP_ERROR",
                        "TRANSPORT_UNKNOWN",
                        "RESPONSE_ENVELOPE_ERROR",
                    }
                    recent_failures = recent_failures + 1 if failed else 0
                    if status in {
                        "MODEL_ID_CHANGED",
                        "SENSITIVE_RESPONSE_WITHHELD",
                    } or item.get("http_status") in (401, 403, 429):
                        stop = f'Stopped on {item.get("http_status") or status}'
                    elif recent_failures >= 3 and not pilot:
                        stop = "Stopped after three consecutive source or provider failures"
                if not stop:
                    for _ in finished:
                        schedule()
                if started and (sum(statuses.values()) % 100 == 0 or not running):
                    print(
                        json.dumps(
                            {
                                "phase": "pilot" if pilot else "run",
                                "completed_this_phase": sum(statuses.values()),
                                "started_this_phase": started,
                                "statuses": dict(statuses),
                            }
                        ),
                        flush=True,
                    )
    return dict(
        status="STOPPED" if stop else "EXECUTED",
        started=started,
        stop_reason=stop,
        statuses=dict(statuses),
        blocked_profiles=sorted(profiles.keys() - admitted.keys()),
    )


def result_rows(rows, history, config):
    all_frames = bool((config or {}).get("frame_settings"))
    admitted = {
        a["profile"]
        for attempts in history.values()
        for a in attempts
        if accepted_pilot(a, config or {})
        and (not all_frames or a.get("frame_report"))
    }
    for row in rows:
        key = row["request_id"]
        attempts = history.get(key, [])
        final = attempts[-1] if attempts else {}
        status = (
            final["status"]
            if attempts
            else (
                "INSPECTION_ONLY"
                if key is None
                else "NOT_ATTEMPTED" if row["profile"] in admitted else "PILOT_REQUIRED"
            )
        )
        answer = final.get("normalized_answer")
        frame_verified = (
            not all_frames
            or final.get("phase") != "pilot"
            or bool(final.get("frame_report"))
        )
        fields = {
            k: row[k]
            for k in (
                "scope",
                "domain",
                "task",
                "input",
                "sample_id",
                "original_video_ID",
                "legacy_sample_id",
                "source",
                "source_manifest_sha256",
                "protocol",
                "source_version",
                "condition_key",
                "parameters",
                "mask",
                "background_id",
                "object_id",
                "object_ids_in_point_order",
                "truth",
                "accuracy_eligible",
                "choices",
                "prompt_sha256",
                "static_case_id",
                "model_input_sha256",
                "frame_indices",
                "request_id",
                "profile",
            )
        }
        yield dict(
            **fields,
            source_video_sha256=row["source_video"]["sha256"],
            requested_model=(config or {}).get("requested_model"),
            returned_model=final.get("returned_model"),
            attempts=len(attempts),
            attempt_sample_id=final.get("sample_id"),
            status=status,
            finish_reason=final.get("finish_reason"),
            raw_answer=final.get("raw_answer"),
            normalized_answer=answer,
            strict_format_valid=final.get("strict_format_valid", False),
            correct=(
                (answer == row["truth"])
                if answer is not None and row["accuracy_eligible"] and frame_verified
                else None
            ),
            http_status=final.get("http_status"),
            response_body_sha256=final.get("response_body_sha256"),
        )


def summarize(results):
    groups = defaultdict(list)
    for row in results:
        groups[row["profile"]].append(row)
    groups["all"] = [row for rows in groups.values() for row in rows]
    report = {}
    for name, rows in sorted(groups.items()):
        unique = {r["request_id"]: r for r in rows if r["request_id"] is not None}
        sections = {}
        for unit, values in (
            ("logical_rows", rows),
            ("independent_requests", list(unique.values())),
        ):
            planned = sum(r["accuracy_eligible"] for r in values)
            scorable = sum(r["correct"] is not None for r in values)
            correct = sum(r["correct"] is True for r in values)
            sections[unit] = dict(
                total=len(values),
                planned_accuracy_eligible=planned,
                observed_scorable=scorable,
                correct=correct,
                strict_format_valid=sum(r["strict_format_valid"] for r in values),
                correct_over_planned=correct / planned if planned else None,
                accuracy_among_scorable=correct / scorable if scorable else None,
                status_counts=dict(Counter(r["status"] for r in values)),
            )
        report[name] = sections
    return report


def results_csv(rows):
    with StringIO(newline="") as stream:
        writer = csv.DictWriter(stream, CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue().encode("utf-8")


def export(out, archive):
    plan, rows = load_plan(out)
    config_path = out / "config.json"
    config = json.loads(config_path.read_bytes()) if config_path.exists() else None
    with (
        closing(open_db(out)) as db,
        TemporaryDirectory(dir=out, prefix=".return-") as scratch,
    ):
        history = records(db)
        stage = Path(scratch)
        normalized = list(result_rows(rows, history, config))
        summary = dict(
            schema=SCHEMA,
            source_plan_sha256=plan["source_plan_sha256"],
            logical_rows=len(rows),
            unique_requests=plan["unique_requests"],
            request_attempts=sum(map(len, history.values())),
            profiles=summarize(normalized),
        )
        with (stage / "attempts.jsonl").open("wb") as dest:
            for raw_receipt, raw_body in db.execute(
                "SELECT details, body FROM attempts ORDER BY request_id, attempt"
            ):
                receipt = json.loads(raw_receipt)
                receipt["response_body_b64"] = (
                    base64.b64encode(raw_body).decode("ascii")
                    if raw_body is not None
                    and not receipt.get("response_body_withheld")
                    else None
                )
                dest.write(release.encoded(receipt))
        with (stage / "results.jsonl").open("wb") as dest:
            for row in normalized:
                dest.write(release.encoded(row))
        (stage / "results.csv").write_bytes(results_csv(normalized))
        (stage / "config.json").write_bytes(release.encoded(config))
        (stage / "summary.json").write_bytes(release.encoded(summary))
        paths = {
            name: (out if name in ("plan.json", "source-plan.jsonl") else stage) / name
            for name in RETURN_FILES
        }
        checksums = {name: file_sha(path) for name, path in paths.items()}
        archive.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=archive.parent, prefix=".provider-zip-") as temp:
            target = Path(temp) / "return.zip"
            with ZipFile(target, "w", ZIP_DEFLATED) as output:
                for name, path in paths.items():
                    output.write(path, name)
                output.writestr("SHA256SUMS.json", release.encoded(checksums))
            target.replace(archive)
        for name in ("summary.json", "results.csv", "results.jsonl", "attempts.jsonl"):
            (stage / name).replace(out / name)
    checksum = file_sha(archive)
    atomic(
        archive.with_suffix(archive.suffix + ".sha256"),
        f"{checksum}  {archive.name}\n".encode(),
    )
    return dict(
        status="EXPORTED",
        archive=str(archive),
        sha256=checksum,
        logical_rows=summary["logical_rows"],
        unique_requests=summary["unique_requests"],
        request_attempts=summary["request_attempts"],
    )


def verify(archive, *, with_history=False):
    checksum = file_sha(archive)
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    if sidecar.exists():
        release.require(
            sidecar.read_text().split()[:1] == [checksum], "Return ZIP SHA-256 mismatch"
        )
    with ZipFile(archive) as bundle:
        files = RETURN_FILES if "results.csv" in bundle.namelist() else LEGACY_RETURN_FILES
        release.require(
            sorted(bundle.namelist()) == sorted((*files, "SHA256SUMS.json")),
            "Unexpected archive members",
        )
        checksums = json.loads(bundle.read("SHA256SUMS.json"))
        release.require(set(checksums) == set(files), "Missing member SHA-256")
        for name, expected in checksums.items():
            with bundle.open(name) as stream:
                release.require(
                    hashlib.file_digest(stream, "sha256").hexdigest() == expected,
                    f"Archive SHA mismatch: {name}",
                )
        plan = json.loads(bundle.read("plan.json"))
        config = json.loads(bundle.read("config.json"))
        rows = load_rows(bundle.read("source-plan.jsonl"))
        selected = release.unique_requests(rows)
        release.require(
            known_schema(plan["schema"])
            and release.known_public_release(plan["revision"], plan["index_sha256"])
            and plan["source_plan_sha256"] == checksums["source-plan.jsonl"]
            and plan["profiles"] == release.counts(rows)
            and plan["logical_rows"] == len(rows)
            and plan["unique_requests"] == len(selected),
            "Archive plan does not reconcile",
        )
        provider = PROVIDER
        if config:
            provider = config["provider"]
            release.require(
                config["source_plan_sha256"] == plan["source_plan_sha256"],
                "Run configuration differs from plan",
            )
            for name, source in (("dataset_revision", "revision"), ("dataset_index_sha256", "index_sha256")):
                release.require(config.get(name, plan[source]) == plan[source], "Run dataset differs from plan")
            if provider in native_video_api.HOSTED:
                native_video_api.validate_config(config)
            elif provider == "vllm":
                native_video_api.vllm_endpoint(config["endpoint"])
            else:
                release.require(
                    provider == PROVIDER and config["endpoint"] == BASE_URL,
                    "Run configuration differs from provider",
                )
            gateway_frames = config.get("video_delivery") == "ordered-images"
            if "video_delivery" in config:
                release.require(
                    provider == PROVIDER
                    and config["video_delivery"] in GATEWAY_VIDEO_DELIVERIES
                    and config.get("token_field") in GATEWAY_TOKEN_FIELDS
                    and config.get("image_detail") in (None, *GATEWAY_IMAGE_DETAILS)
                    and "preset" not in config
                    and (
                        config.get("pilot_admission") is None if "frame_settings" in config
                        else config.get("pilot_admission") == "completion_envelope"
                    ),
                    "Unexpected NVIDIA Gateway configuration",
                )
            if gateway_frames:
                release.require(
                    "video_prompt_prefix" in config
                    and config.get("video_delivery_settings") == native_video_api.ordered_image_settings(PROVIDER),
                    "Ordered image configuration differs",
                )
            if "video_prompt_prefix" in config:
                release.require(
                    (gateway_frames or (provider in native_video_api.ORDERED_IMAGES and "frame_settings" in config))
                    and config["video_prompt_prefix"] == ORDERED_IMAGE_VIDEO_PROMPT_PREFIX,
                    "Unexpected video prompt prefix",
                )
            if "preset" in config:
                release.require(
                    config["preset"] == NVIDIA_REPEAT_PRESET
                    and provider == PROVIDER
                    and config.get("generation_parameters") == NVIDIA_REPEAT_GENERATION,
                    "NVIDIA configuration differs from the repeat preset",
                )
            if "frame_settings" in config:
                release.require(
                    provider in {PROVIDER, "vllm", *native_video_api.FRAME_LISTS}
                    and config["frame_settings"] == requested_frame_settings(config),
                    "All-frame configuration differs",
                )
        history = defaultdict(list)
        changed_models = set()
        admitted_models = set()
        full_frame_profiles = set()
        full_frame_runs = set()
        frame_deployments = set()
        with bundle.open("attempts.jsonl") as stream:
            for line in stream:
                entry = json.loads(line)
                key = entry["request_id"]
                body = entry.pop("response_body_b64")
                release.require(
                    key in selected
                    and entry["attempt"] == len(history[key]) + 1
                    and entry["attempt"] <= 2
                    and entry["sample_id"] == selected[key]["sample_id"]
                    and entry["profile"] == selected[key]["profile"],
                    "Invalid attempt identity",
                )
                if "request_sha256" in entry:
                    if "video_prompt_prefix" in config or "video_delivery" in config or config.get("preset") == NVIDIA_REPEAT_PRESET:
                        expected_prompt = request_prompt(selected[key], config)
                        release.require(
                            entry.get("request_prompt_sha256") == release.sha(expected_prompt.encode("utf-8")),
                            "Request prompt SHA-256 differs from the recorded text",
                        )
                    source = selected[key]["media"]
                    verified = entry["verified_media"]
                    release.require(
                        bool(re.fullmatch(r"[0-9a-f]{64}", entry["request_sha256"]))
                        and len(source) == len(verified)
                        and all(
                            {k: v for k, v in actual.items() if k not in {"bytes", "ordered_frames"}} == planned
                            and actual["bytes"] > 0
                            for planned, actual in zip(source, verified)
                        ),
                        "Sent media receipt differs from frozen input",
                    )
                    if "request_bytes" in entry:
                        size = entry["request_bytes"]
                        release.require(type(size) is int and size > 0, "Invalid request size")
                        native_video_api.validate_payload_size(provider, (config or {}).get("endpoint"), size)
                    if provider in native_video_api.FRAME_LISTS or config.get("video_delivery") == "ordered-images":
                        for actual in verified:
                            if actual["mime"] == "video/mp4":
                                wire = actual["ordered_frames"]
                                release.require(
                                    wire["count"] == 180 and wire["fps"] == 60
                                    and bool(re.fullmatch(r"[0-9a-f]{64}", wire["rgb_sha256"]))
                                    and wire["rgb_sha256"] == actual.get(
                                        "processor_input_rgb_sha256", wire["rgb_sha256"]
                                    ),
                                    "Ordered frame receipt differs from frozen input",
                                )
                if body is not None:
                    content = base64.b64decode(body, validate=True)
                    release.require(
                        entry["response_body_sha256"] == release.sha(content)
                        and entry["response_bytes"] == len(content),
                        "Raw response identity differs",
                    )
                    parsed = response_fields(
                        content,
                        entry["http_status"],
                        selected[key]["choices"],
                        None,
                        provider,
                    )
                    if entry["status"] == "MODEL_ID_CHANGED":
                        changed_models.add(parsed["returned_model"])
                        parsed.update(
                            status="MODEL_ID_CHANGED",
                            normalized_answer=None,
                            strict_format_valid=False,
                            modality_admitted=False,
                        )
                    release.require(
                        all(entry.get(k) == v for k, v in parsed.items()),
                        "Recorded answer differs from original response",
                    )
                else:
                    withheld = entry.get("response_body_withheld")
                    release.require(
                        "response_body_sha256" not in entry
                        or (
                            entry["status"] == "SENSITIVE_RESPONSE_WITHHELD"
                            and withheld in {"credential_echo", "media_echo"}
                        ),
                        "Original response body is missing",
                    )
                if accepted_pilot(entry, config or {}):
                    admitted_models.add(entry["returned_model"])
                if "frame_report" in entry:
                    release.require(
                        bool((config or {}).get("frame_settings")),
                        "Unexpected server frame report",
                    )
                    validate_frame_report(entry["frame_report"], entry, config)
                    full_frame_profiles.add(entry["profile"])
                    frame_deployments.add(entry["frame_report"]["deployment"])
                if entry["phase"] == "run" and (config or {}).get("frame_settings"):
                    full_frame_runs.add(entry["profile"])
                history[key].append(entry)
        release.require(
            full_frame_runs <= full_frame_profiles and len(frame_deployments) <= 1,
            "All-frame batch evidence does not reconcile",
        )
        release.require(
            (not changed_models or bool(admitted_models))
            and not changed_models & admitted_models
            and len(admitted_models) <= 1,
            "Provider model identities do not reconcile",
        )
        release.require(
            not history or config is not None, "Attempts have no configuration"
        )
        expected = list(result_rows(rows, history, config))
        if "results.csv" in files:
            release.require(bundle.read("results.csv") == results_csv(expected), "CSV differs from rescored rows")
        with bundle.open("results.jsonl") as stream:
            for recorded, computed in zip(stream, expected, strict=True):
                release.require(
                    json.loads(recorded) == computed,
                    "Normalized row differs from attempts",
                )
        summary = json.loads(bundle.read("summary.json"))
        release.require(
            known_schema(summary["schema"])
            and summary["source_plan_sha256"] == plan["source_plan_sha256"]
            and summary["profiles"] == summarize(expected)
            and summary["logical_rows"] == len(rows)
            and summary["unique_requests"] == len(selected)
            and summary["request_attempts"] == sum(map(len, history.values())),
            "Result counts do not reconcile",
        )
    result = dict(
        status="VERIFIED",
        sha256=checksum,
        logical_rows=len(rows),
        unique_requests=len(selected),
        request_attempts=summary["request_attempts"],
    )
    return (result, config, history) if with_history else result


def compare(archive, against):
    first, first_config, first_history = verify(archive, with_history=True)
    second, second_config, second_history = verify(against, with_history=True)
    release.require(first["sha256"] != second["sha256"], "Use two independent return archives")
    release.require(
        first_config and first_config == second_config,
        "Compare two runs with the same selection, model and request settings",
    )
    common = first_history.keys() & second_history.keys()
    raw_pairs = label_pairs = raw_changes = label_flips = 0
    changes = []
    identity_changes = []
    for key in sorted(common):
        before, after = first_history[key][-1], second_history[key][-1]
        if not (before.get("request_sha256") and after.get("request_sha256")):
            continue
        release.require(
            before["request_sha256"] == after["request_sha256"],
            f"Request bytes differ for {before['sample_id']}",
        )
        if not all(isinstance(entry.get("raw_answer"), str) for entry in (before, after)):
            continue
        if any(
            before.get(field) is not None and after.get(field) is not None
            and before[field] != after[field]
            for field in ("returned_model", "system_fingerprint")
        ) or (
            before.get("frame_report") and after.get("frame_report")
            and before["frame_report"]["deployment"] != after["frame_report"]["deployment"]
        ):
            identity_changes.append(before["sample_id"])
            continue
        raw_pairs += 1
        raw_changed = before["raw_answer"] != after["raw_answer"]
        raw_changes += raw_changed
        old, new = before["normalized_answer"], after["normalized_answer"]
        both_labels = old is not None and new is not None
        label_pairs += both_labels
        flipped = both_labels and old != new
        label_flips += flipped
        if raw_changed or flipped:
            changes.append(dict(
                sample_id=before["sample_id"], raw_changed=raw_changed,
                before=old, after=new, label_flipped=flipped,
            ))
    return dict(
        status="COMPARED", planned_requests=first["unique_requests"],
        common_attempted_requests=len(common), paired_raw_answers=raw_pairs,
        paired_labels=label_pairs, raw_changes=raw_changes, label_flips=label_flips,
        changes=changes, changed_server_identity=identity_changes,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("plan", "pilot", "run", "evaluate", "export", "verify", "compare")
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--against", type=Path, help="Second return ZIP for a repeat comparison")
    parser.add_argument(
        "--preset", choices=(NVIDIA_REPEAT_PRESET,),
        help="Request temperature 0 and a fixed seed on the NVIDIA gateway",
    )
    parser.add_argument("--data-root", type=Path, help="Dataset cache for downloaded inputs")
    parser.add_argument(
        "--metadata-root", type=Path, help="Offline root of the pinned public manifests"
    )
    parser.add_argument(
        "--media-root",
        type=Path,
        help="Use only media already in this local directory",
    )
    parser.add_argument(
        "--model",
        help="Exact model ID, e.g. gpt-6-astra, claude-opus-5-5 or gemini-4-argon",
    )
    parser.add_argument(
        "--provider",
        choices=(PROVIDER, "nvidia", "vllm", *native_video_api.HOSTED),
        metavar="PROVIDER",
        help="Override the model's route (nvidia, vllm, astra, claude, gemini or qwen-frames).",
    )
    parser.add_argument(
        "--video-delivery", choices=GATEWAY_VIDEO_DELIVERIES,
        help="NVIDIA Gateway video transport (default: native).",
    )
    parser.add_argument(
        "--image-detail", choices=GATEWAY_IMAGE_DETAILS,
        help="Send an explicit image detail only when the Gateway model supports it.",
    )
    parser.add_argument(
        "--token-field", choices=GATEWAY_TOKEN_FIELDS,
        help="NVIDIA Gateway output token field (default: max_completion_tokens).",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--reasoning-mode", choices=("standard", "pro"))
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="Maximum new attempts in this invocation, including pilots",
    )
    budget.add_argument(
        "--all-requests", action="store_true",
        help="Authorize every unique request in the selected task or range",
    )
    parser.add_argument(
        "--concurrency", "--workers", dest="workers", type=int,
        help="Parallel calls after the pilot, from 1 to 16 (default: 2; nvidia-repeat: 1)",
    )
    parser.add_argument("--scope", action="append", choices=release.SCOPES)
    parser.add_argument(
        "--task",
        action="append",
        choices=TASK_SCOPES,
        help="Select a task and domain, e.g. t1-classic; defaults to all-frame video",
    )
    parser.add_argument("--input", choices=("video", "static"))
    sample = parser.add_mutually_exclusive_group()
    sample.add_argument(
        "--sample", action="append",
        help="Exact published trial ID, video filename or dataset-relative video path. Repeat to select several trials.",
    )
    sample.add_argument("--sample-id", action="append", help=argparse.SUPPRESS)
    parser.add_argument("--from-index", type=int, help="First trial index, inclusive (zero-based)")
    parser.add_argument("--to-index", type=int, help="Stop before this trial index")
    frames = parser.add_mutually_exclusive_group()
    frames.add_argument(
        "--all-frames",
        action="store_true",
        help="Request original frames and gate bulk on a server report",
    )
    frames.add_argument(
        "--sampled-video",
        action="store_true",
        help="Use the provider sampling path as a separate comparison",
    )
    parser.add_argument(
        "--frame-report",
        type=Path,
        help="Server decoder and processor evidence from the pilot",
    )
    parser.add_argument(
        "--retry",
        action="append",
        help="Explicit request ID to try once more; never re-asks a returned answer",
    )
    parser.add_argument(
        "--retry-uncertain",
        action="store_true",
        help="A previous request may already have reached the provider",
    )
    parser.add_argument(
        "--pilot-trial",
        action="append",
        help="Choose another previously unasked request ID for a blocked profile",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Authorize real calls from this shell"
    )
    args = parser.parse_args(argv)
    if args.provider == "nvidia":
        args.provider = PROVIDER
    if args.preset == NVIDIA_REPEAT_PRESET:
        args.provider = args.provider or PROVIDER
        if args.provider != PROVIDER or args.command in {"verify", "export", "compare"}:
            parser.error("nvidia-repeat applies to NVIDIA Inference Gateway evaluation")
        if any(value is not None for value in (args.temperature, args.top_p, args.seed)) or args.disable_thinking:
            parser.error("nvidia-repeat fixes temperature, top-p and seed. Omit manual generation flags.")
        args.input = args.input or "video"
        args.model = args.model or os.environ.get("NVIDIA_VIDEO_MODEL_ID")
    if args.workers is None:
        args.workers = 1 if args.preset else 2
    auto_provider = None
    if args.provider is None:
        auto_provider = native_video_api.model_provider(args.model) if args.model else None
        if (
            args.model and auto_provider is None and "/" not in args.model
            and args.command not in {"export", "verify", "compare"}
        ):
            parser.error("Unknown model route. Set --provider explicitly for this model.")
        args.provider = auto_provider or PROVIDER
    args.model = args.model or os.environ.get("VLM_API_MODEL")
    if args.preset and not args.model and args.command != "plan":
        parser.error("Set --model or NVIDIA_VIDEO_MODEL_ID to the exact served model ID.")
    if any((args.video_delivery, args.image_detail, args.token_field)) and (
        args.provider != PROVIDER or args.preset
    ):
        parser.error("These transport options apply to NVIDIA Gateway without --preset.")
    if args.video_delivery == "ordered-images" and args.sampled_video:
        parser.error("--sampled-video uses native video delivery.")
    args.download_metadata = bool(args.data_root and args.metadata_root is None)
    args.download_media = bool(args.data_root and args.media_root is None)
    if args.data_root:
        args.metadata_root = args.metadata_root or args.data_root.expanduser()
        args.media_root = args.media_root or args.data_root.expanduser()
    generation = NVIDIA_REPEAT_GENERATION if args.preset == NVIDIA_REPEAT_PRESET else {}
    if args.top_p is not None:
        if args.provider != PROVIDER or not 0 < args.top_p <= 1:
            parser.error("Use --top-p in (0, 1] with NVIDIA Gateway.")
        generation["top_p"] = args.top_p
    if args.temperature is not None or args.seed is not None or args.disable_thinking:
        if args.provider not in {PROVIDER, "vllm", "qwen", "qwen-frames"}:
            parser.error("These generation settings require NVIDIA Gateway, vllm or Qwen")
        if args.provider == PROVIDER and args.disable_thinking:
            parser.error("--disable-thinking requires vllm or Qwen")
        if args.temperature is not None:
            if not 0 <= args.temperature < 2:
                parser.error("--temperature must be in [0, 2)")
            generation["temperature"] = args.temperature
        if args.seed is not None:
            if not 0 <= args.seed < 2**31:
                parser.error("--seed must be in [0, 2**31)")
            generation["seed"] = args.seed
        if args.disable_thinking:
            if args.provider == "vllm":
                generation["chat_template_kwargs"] = {"enable_thinking": False}
            else:
                generation["enable_thinking"] = False
    if args.reasoning_effort or args.reasoning_mode:
        if args.provider == PROVIDER and args.reasoning_effort and not (args.preset or args.reasoning_mode):
            generation["reasoning_effort"] = args.reasoning_effort
        elif args.provider == "astra":
            generation["reasoning"] = {"effort": args.reasoning_effort or "medium"}
            if args.reasoning_mode:
                generation["reasoning"]["mode"] = args.reasoning_mode
        else:
            parser.error("Use --reasoning-effort with Astra or NVIDIA Gateway; --reasoning-mode requires Astra.")
    args.generation_parameters = generation
    args.selection_requested = bool(
        args.task or args.scope or args.input or args.sample or args.sample_id
        or args.from_index is not None or args.to_index is not None
    )
    if args.task:
        args.scope = list(
            dict.fromkeys(
                [*(args.scope or []), *(TASK_SCOPES[task] for task in args.task)]
            )
        )
    if args.scope or args.sample or args.data_root or auto_provider:
        args.input = args.input or "video"
    args.index_bounds = None
    if args.from_index is not None or args.to_index is not None:
        if (
            len(args.scope or []) != 1 or not args.input or args.sample_id or args.sample
        ):
            parser.error("Index selection requires one public --task and one modality, without --sample")
        args.index_bounds = (args.from_index, args.to_index)
    if args.provider != PROVIDER and (
        args.task or args.data_root or auto_provider or args.provider in native_video_api.FRAME_LISTS
    ):
        args.all_frames = args.all_frames or (
            args.input == "video" and not args.sampled_video
        )
    try:
        if args.command == "verify":
            release.require(args.archive, "Set --archive for verification")
            result = verify(args.archive)
        elif args.command == "compare":
            release.require(args.archive and args.against, "Set --archive and --against for comparison")
            result = compare(args.archive, args.against)
        else:
            release.require(args.out, "Set --out to a run directory on local storage")
            release.require(
                1 <= args.workers <= 16 and args.max_tokens > 0,
                "Invalid workers or token limit",
            )
            args.out.mkdir(parents=True, exist_ok=True, mode=0o700)
            with (args.out / ".run.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = command(args)
    except (OSError, ValueError, KeyError, sqlite3.Error, BadZipFile) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if result.get("stop_reason") or result.get("status") == "PARTIAL":
        raise SystemExit(2)


def command(args):
    archive = args.archive or args.out / "return.zip"
    if args.command == "export":
        return export(args.out, archive)
    all_requests = getattr(args, "all_requests", False)
    if args.command != "plan":
        release.require(
            args.execute and (all_requests or args.max_requests > 0),
            "Set --execute and either --all-requests or a positive --max-requests",
        )
    if args.command == "evaluate":
        release.require(not args.retry, "Use pilot or run for an explicit retry")
    if args.command != "pilot":
        release.require(not args.pilot_trial, "--pilot-trial applies to pilot only")
    index_bounds = getattr(args, "index_bounds", None)
    if args.command == "plan" or (
        args.command == "evaluate" and (
            not (args.out / "plan.json").exists()
            or getattr(args, "selection_requested", False)
        )
    ):
        local_selection = (
            getattr(args, "selection_requested", False)
            or getattr(args, "task", None) or getattr(args, "data_root", None) or index_bounds
        )
        plan, rows = build_plan(
            args.out, args.metadata_root,
            scopes=args.scope if local_selection else None,
            modality=args.input if local_selection else None,
            samples=getattr(args, "sample_id", None),
            selectors=getattr(args, "sample", None),
            index_bounds=index_bounds,
            download_missing=getattr(args, "download_metadata", False),
        )
    else:
        plan, rows = load_plan(args.out)
        if index_bounds is not None:
            selection = plan.get("index_selection")
            release.require(
                selection and selection == index_selection(
                    index_bounds, args.scope[0], args.input, selection["total_trials"],
                ),
                "This output directory has a different index selection. Use a new --out.",
            )
    if args.command == "plan":
        return {
            "status": "PLANNED_NO_MODEL_CALLS",
            **plan,
            "selected_requests": len(selected_requests(rows, args)),
            **(
                {"provider": args.provider, "requested_model": args.model}
                if args.model else {}
            ),
        }
    if all_requests:
        args.max_requests = len(selected_requests(rows, args))
    if args.command in ("pilot", "run"):
        return execution(
            args.out, plan, rows, args, pilot=args.command == "pilot"
        )
    pilot = execution(args.out, plan, rows, args, pilot=True)
    args.max_requests -= pilot["started"]
    run = None
    if args.max_requests > 0 and not pilot.get("stop_reason"):
        run = execution(args.out, plan, rows, args, pilot=False)
    packaged = export(args.out, archive)
    with closing(open_db(args.out)) as db:
        histories = records(db)
    selected = selected_requests(rows, args)
    pending_frame_profiles = set()
    if args.all_frames:
        profiles = {row["profile"] for row in selected.values()}
        admitted_pilots = [
            attempt for attempts in histories.values() for attempt in attempts
            if attempt["profile"] in profiles
            and attempt["phase"] == "pilot"
            and attempt.get("modality_admitted")
        ]
        pending_frame_profiles = {
            attempt["profile"] for attempt in admitted_pilots
        } - {attempt["profile"] for attempt in admitted_pilots if attempt.get("frame_report")}
    finished = sum(
        1
        for key in selected
        if histories.get(key)
        and histories[key][-1]["status"]
        in {
            "ANSWERED",
            "FORMAT_OR_TOOL_FAILURE",
            "TRUNCATED",
        }
        and (
            not args.all_frames
            or histories[key][-1]["phase"] != "pilot"
            or histories[key][-1].get("frame_report")
        )
    )
    return {
        **packaged,
        "status": "COMPLETE" if finished == len(selected) else "PARTIAL",
        "selected_requests": len(selected),
        "finished_selected_requests": finished,
        "awaiting_server_frame_report": sorted(pending_frame_profiles),
        "pilot": pilot,
        "run": run,
    }


if __name__ == "__main__":
    main()
