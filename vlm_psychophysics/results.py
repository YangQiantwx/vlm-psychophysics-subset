"""Score independent inputs and export a self-contained return archive."""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import io
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
import zipfile

from . import data


REPORT_FILES = ("results.jsonl", "results.csv", "summary.json", "curves.png")
PROOF_FILES = (
    "evidence/run.json",
    "evidence/config.json",
    "evidence/runtime-lock.json",
    "evidence/input-manifest.jsonl",
    "evidence/source-code.json",
    "evidence/execution.json",
)
SESSION_PATTERN = (
    r"evidence/sessions/GPU-[0-9a-fA-F-]{36}/[a-z0-9-]+/[0-9]+-[0-9]+/"
    r"(?:resolved-model\.json|runtime\.json|input-references\.jsonl|"
    r"input-observations\.jsonl|returned\.jsonl|input-failures\.jsonl|exit\.json|process-exit\.json)"
)
CSV_COLUMNS = (
    "job_id",
    "model",
    "recipe",
    "task",
    "domain",
    "modality",
    "sample_id",
    "status",
    "raw_answer",
    "prediction",
    "truth",
    "accuracy_eligible",
    "correct",
    "gpu_uuid",
    "model_input_sha256",
    "prompt_sha256",
    "config_sha256",
    "alias_count",
    "condition",
    "finish_reason",
)
CONDITION_LABELS = {
    "T1": "Modulation depth",
    "T2": "Luminance increment",
    "T3": "Angle (degrees)",
    "T4": "Coherence",
    "T5": "Event count",
}


def write_json(path, value):
    write_bytes(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode()
        + b"\n",
    )


def write_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, prefix=".writing-", delete=False
    ) as temporary:
        temp_path = Path(temporary.name)
        try:
            temporary.write(payload)
            temporary.flush()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)


def job_id(case, model):
    return data.digest([model, case["case_id"]])


def recipe_for(config, case, model):
    key = data.profile_key(case["modality"], data.case_scope(case))
    return data.recipe_name(
        config["models"][model]["profiles"][key], case.get("cohort")
    )


def condition_for(task, modality, parameters):
    """Keep the manipulated conditions and exclude temporal labels from stills."""
    if task == "T1":
        condition = {
            "value": parameters.get("modulation_depth", parameters.get("m")),
            "spatial_frequency": parameters.get(
                "sf_cycles_per_aperture", parameters.get("sf")
            ),
        }
    elif task == "T2":
        condition = {"value": parameters.get("delta_Y")}
        if "background_Y" in parameters:
            condition["background_Y"] = parameters["background_Y"]
        if (
            modality == "video"
            and "onset_frame" in parameters
            and "offset_frame" in parameters
        ):
            condition["duration_frames"] = (
                parameters["offset_frame"] - parameters["onset_frame"]
            )
    elif task == "T3":
        condition = {"value": parameters.get("angle_deg", parameters.get("angle"))}
    elif task == "T4":
        condition = {
            "value": parameters.get("coherence"),
            "element_count": parameters.get("N"),
        }
        if "appearance" in parameters:
            condition["appearance"] = parameters["appearance"]
    else:
        condition = {"value": len(parameters["events"])}
    if "group" in parameters:
        condition["stimulus_group"] = parameters["group"]
    return condition


def case_condition(case):
    conditions = {
        data.json_bytes(
            condition_for(case["task"], case["modality"], alias["parameters"])
        )
        for alias in case["aliases"]
    }
    if len(conditions) != 1:
        raise ValueError(
            f'Identical static input crosses plotted conditions: {case["sample_id"]}'
        )
    return json.loads(conditions.pop())


def score(raw_answer, choices, truth, eligible, state, technical_pass):
    """Use exact labels after trimming only the returned answer's whitespace."""
    stripped = raw_answer.strip() if isinstance(raw_answer, str) else None
    prediction = stripped if stripped in choices else None
    if state == "PENDING":
        status = "MISSING"
    elif state == "INTENT":
        status = "UNCERTAIN"
    elif state != "RETURNED":
        raise ValueError(f"Unknown execution state {state!r}")
    elif technical_pass is not True:
        status = "TECHNICAL_FAILURE"
    elif prediction is None:
        status = "INVALID_ANSWER"
    else:
        status = "SCORED" if eligible else "UNSCORED"
    correct = (status == "SCORED" and prediction == truth) if eligible else None
    return prediction, status, correct


def result_row(case, model, config, config_sha, attempt):
    answer = attempt.get("answer") or {}
    state = attempt.get("state", "PENDING")
    prediction, status, correct = score(
        answer.get("raw_answer"),
        case["choices"],
        case["truth"],
        case["accuracy_eligible"],
        state,
        answer.get("technical_pass"),
    )
    return {
        "job_id": job_id(case, model),
        "case_id": case["case_id"],
        "sample_id": case["sample_id"],
        "model": model,
        "recipe": recipe_for(config, case, model),
        "recipe_source_sha256": (
            config["models"][model]["equal_canvas_single_source_profile"]
            if case.get("cohort") == "single"
            else config["models"][model]["source_profiles"][
                data.profile_key(case["modality"], data.case_scope(case))
            ]
        )["sha256"],
        "task": case["task"],
        "domain": case["domain"],
        "modality": case["modality"],
        "truth": case["truth"],
        "choices": case["choices"],
        "accuracy_eligible": case["accuracy_eligible"],
        "condition": case_condition(case),
        "alias_count": len(case["aliases"]),
        "model_input_sha256": case["model_input_sha256"],
        "prompt_sha256": case["prompt_sha256"],
        "config_sha256": config_sha,
        "gpu_uuid": attempt.get("gpu_uuid"),
        "state": state,
        "status": status,
        "prediction": prediction,
        "correct": correct,
        "raw_answer": answer.get("raw_answer"),
        "finish_reason": answer.get("finish_reason"),
        "stop_reason": answer.get("stop_reason"),
        "technical_pass": answer.get("technical_pass"),
        "technical_error": answer.get("technical_error"),
        "profile_sha256": answer.get("profile_sha256"),
        "input_receipt": answer.get("input_receipt"),
        "prompt_token_ids": answer.get("prompt_token_ids"),
        "prompt_token_ids_sha256": answer.get("prompt_token_ids_sha256"),
        "prompt_token_count": answer.get("prompt_token_count"),
        "output_token_ids": answer.get("output_token_ids"),
        "error": attempt.get("error"),
        "started_ns": attempt.get("started_ns"),
        "finished_ns": attempt.get("finished_ns"),
    }


def summarize(rows):
    groups = defaultdict(lambda: [0] * 9)
    overall = [0] * 9
    for row in rows:
        key = (row["task"], row["domain"], row["model"], row["modality"])
        record = groups[(*key, data.json_bytes(row["condition"]))]
        status = row["status"]
        counts = (
            1,
            row["accuracy_eligible"],
            row["correct"] is True,
            status == "SCORED",
            status == "MISSING",
            status == "UNCERTAIN",
            status == "INVALID_ANSWER",
            status == "TECHNICAL_FAILURE",
            status == "UNSCORED",
        )
        for group in (record, overall):
            for i, count in enumerate(counts):
                group[i] += count

    def report(counts):
        (
            planned,
            eligible,
            correct,
            scored,
            missing,
            uncertain,
            invalid,
            technical,
            unscored,
        ) = counts
        return {
            "planned": planned,
            "accuracy_denominator": eligible,
            "correct": correct,
            "scored": scored,
            "missing": missing,
            "uncertain": uncertain,
            "invalid_answers": invalid,
            "technical_failures": technical,
            "unscored": unscored,
            "accuracy": correct / eligible if eligible else None,
        }

    def grouped_reports(table, condition_key, decode):
        return [
            dict(
                task=task,
                domain=domain,
                model=model,
                modality=modality,
                **{condition_key: decode(condition)},
                **report(counts),
            )
            for (task, domain, model, modality, condition), counts in sorted(
                table.items(), key=lambda item: (*item[0][:4], str(item[0][4]))
            )
        ]

    return {
        "overall": report(overall),
        "conditions": grouped_reports(groups, "condition", json.loads),
    }


def draw_curves(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = defaultdict(list)
    for point in summary["conditions"]:
        value = point["condition"]["value"]
        if (
            value is None
            or point["accuracy"] is None
            or point["missing"]
            or point["uncertain"]
        ):
            continue
        facets = tuple(
            (key, setting)
            for key, setting in point["condition"].items()
            if key != "value"
        )
        panels[(point["task"], point["domain"], point["modality"], facets)].append(
            point
        )
    columns = min(4, max(1, len(panels)))
    height = max(1, (len(panels) + columns - 1) // columns)
    figure, axes = plt.subplots(
        height, columns, squeeze=False, figsize=(3.7 * columns, 2.8 * height)
    )
    labels = {}
    facet_names = {
        "spatial_frequency": "SF",
        "background_Y": "Bg Y",
        "duration_frames": "Frames",
        "element_count": "N",
        "appearance": "Appearance",
        "stimulus_group": "Group",
    }
    for panel, axis in zip(sorted(panels), axes.flat):
        task, domain, modality, facets = panel
        for index, model in enumerate(data.MODELS):
            series = sorted(
                (p["condition"]["value"], p["accuracy"])
                for p in panels[panel]
                if p["model"] == model
            )
            if series:
                x, y = zip(*series)
                (line,) = axis.plot(
                    x, y, marker="o", markersize=3, color=f"C{index}", label=model
                )
                labels[model] = line
        axis.axhline(0.5, linewidth=0.7, linestyle=":", color="0.55")
        title = f"{task} · {domain} · {modality}"
        if facets:
            title += "\n" + " · ".join(
                f"{facet_names[key]} {value}" for key, value in facets
            )
        axis.set_title(title, fontsize=9)
        axis.set_xlabel(CONDITION_LABELS[task])
        if task == "T5":
            axis.set_xticks([1, 2])
            axis.set_xlim(0.8, 2.2)
        axis.set_ylim(-0.02, 1.02)
        axis.set_ylabel("Accuracy")
    for axis in list(axes.flat)[len(panels) :]:
        axis.set_axis_off()
    if panels:
        figure.legend(
            labels.values(), labels.keys(), loc="lower center", ncol=len(labels)
        )
        figure.tight_layout(rect=(0, 0.07 / height, 1, 1))
    else:
        figure.text(0.5, 0.5, "No completed accuracy-eligible calls", ha="center")
    try:
        figure.savefig(output, dpi=150)
    finally:
        plt.close(figure)


def csv_rows(stream, rows):
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        fields = {key: row[key] for key in CSV_COLUMNS}
        fields["condition"] = data.json_bytes(fields["condition"]).decode()
        writer.writerow(fields)


def jsonl_bytes(rows):
    return b"".join(data.json_bytes(row) + b"\n" for row in rows)


def write_reports(out, rows, execution):
    write_bytes(out / "results.jsonl", jsonl_bytes(rows))
    csv_output = io.StringIO(newline="")
    csv_rows(csv_output, rows)
    write_bytes(out / "results.csv", csv_output.getvalue().encode())
    summary = summarize(rows)
    write_json(out / "summary.json", summary)
    draw_curves(summary, out / "curves.png")
    write_json(out / "evidence/execution.json", execution)
    return summary


def session_files(out, name=None):
    root = out / "evidence/sessions"
    return sorted(
        path
        for path in root.glob(f"*/*/*/{name or '*'}")
        if path.is_file()
        and re.fullmatch(SESSION_PATTERN, path.relative_to(out).as_posix())
    )


def make_archive(out):
    sessions = [path.relative_to(out).as_posix() for path in session_files(out)]
    members = (*PROOF_FILES, *REPORT_FILES, *sessions)
    hashes = {name: data.file_sha(out / name) for name in members}
    with NamedTemporaryFile(dir=out, prefix=".return-", delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(
            temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3
        ) as archive:
            for name in members:
                archive.write(out / name, name)
            archive.writestr("SHA256SUMS.json", data.json_bytes(hashes) + b"\n")
        archive_path = out / "return.zip"
        temp_path.replace(archive_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return archive_path


def validate_case(case):
    """Check request identity independently of scoring."""
    task = case["task"]
    modality = case["modality"]
    prompt = case["prompt"]
    paired = task == "T5"
    scope = data.case_scope(case)
    if (
        scope
        not in (*data.SCOPES, *data.PAIRED_SCOPES, *data.paired_inputs.LEGACY_SCOPES)
        or modality not in data.MODALITIES
        or not isinstance(prompt, str)
        or not prompt
    ):
        raise ValueError("Unexpected task, modality, choices or prompt")
    if data.paired_inputs.is_current_reference(scope):
        counts = {len(alias["parameters"]["events"]) for alias in case["aliases"]}
        expected = {"single": 1, "dual": 2}.get(case["cohort"])
        if counts != {expected}:
            raise ValueError(f'Unexpected T5 cohort for {case["sample_id"]}')
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != case["prompt_sha256"]:
        raise ValueError(f'Prompt SHA-256 differs for {case["sample_id"]}')
    if case["video_sha256"] != data.require_sha(case["video"]["sha256"]):
        raise ValueError(f'Source video SHA-256 differs for {case["sample_id"]}')
    images = case["images"]
    if modality == "static":
        indices = [image["frame_index"] for image in images]
        if indices != case["frame_indices"]:
            raise ValueError(f'Static image order differs for {case["sample_id"]}')
        frames = indices[1:] if paired else indices
        identity = data.static_identity(
            scope, task, images, frames, prompt, case["choices"], case["sample_id"]
        )
        if paired and case["prompt_sha256"] != data.paired_inputs.STATIC_PROMPT_SHA256:
            raise ValueError(f'Unexpected T5 static prompt for {case["sample_id"]}')
        for alias in case["aliases"]:
            frames = alias["frame_indices"]
            expected = data.static_frame_indices({**alias, "task": task})
            if (
                frames != expected
                or data.static_identity(
                    scope,
                    task,
                    alias["images"],
                    frames,
                    prompt,
                    case["choices"],
                    alias["sample_id"],
                )
                != identity
            ):
                raise ValueError(f'Static alias input differs for {alias["sample_id"]}')
    else:
        if len(images) != int(paired):
            raise ValueError(f'Unexpected video reference for {case["sample_id"]}')
        if paired:
            reference = images[0]
            data.current_release.payload_path(scope, reference["path"])
            data.require_sha(reference["sha256"])
            data.require_sha(reference["pixel_sha256"])
            expected_size = (
                (768, 512)
                if data.paired_inputs.is_current_reference(scope)
                else (160, 160)
            )
            if (
                reference["frame_index"] is not None
                or case["frame_indices"] != [None]
                or data.image_size(reference) != expected_size
            ):
                raise ValueError(f'Unexpected reference frame for {case["sample_id"]}')
        media_identity = data.video_input_identity(case["video_sha256"], images)
        identity = data.digest([media_identity, prompt, case["choices"]])
    case_key = data.digest([scope, modality, case["sample_id"], identity])
    if case["model_input_sha256"] != identity or case["case_id"] != case_key:
        raise ValueError(f'Model input identity differs for {case["sample_id"]}')
    if (
        not case["aliases"]
        or len({alias["sample_id"] for alias in case["aliases"]})
        != len(case["aliases"])
        or any(alias["truth"] != case["truth"] for alias in case["aliases"])
    ):
        raise ValueError(f'Alias/ground truth mismatch for {case["sample_id"]}')
    data.checked_labels(case, case["choices"])


def read_session_proof(archive, members):
    """Index the persisted engine returns and matching CPU/GPU input evidence."""
    session_paths = {
        name.rsplit("/", 1)[0]
        for name in members
        if name.startswith("evidence/sessions/")
    }
    returns = {}
    references = set()
    observations = set()
    profiles = {}
    for session in sorted(session_paths):
        _, _, gpu, recipe, _ = session.split("/")
        resolved = f"{session}/resolved-model.json"
        if resolved in members:
            document = json.loads(archive.read(resolved))
            profile = document["profile"]
            sha = data.digest(profile)
            if (
                sha != document["profile_sha256"]
                or document["gpu_uuid"] != gpu
                or profile["recipe"] != recipe
            ):
                raise ValueError(f"Resolved model identity differs in {session}")
            if f"{session}/runtime.json" in members:
                profiles[session] = sha
        for filename in (
            "returned.jsonl",
            "input-references.jsonl",
            "input-observations.jsonl",
        ):
            name = f"{session}/{filename}"
            if name not in members:
                continue
            with archive.open(name) as stream:
                for line in stream:
                    if not line.endswith(b"\n"):
                        break
                    record = json.loads(line)
                    key = record["reference_key"]
                    if filename == "returned.jsonl":
                        model = recipe.split("-", 1)[0]
                        job = data.digest([model, record["case_id"]])
                        if job in returns:
                            raise ValueError(
                                f"Duplicate durable engine return for {job}"
                            )
                        returns[job] = (session, gpu, recipe, record)
                    elif filename == "input-references.jsonl":
                        ids = record["reference"]["prompt_token_ids"]
                        references.add((session, key, data.digest(ids), len(ids)))
                    elif record["GPU"] and record["application"]:
                        observations.add((session, key, record["sample_id"]))
    return returns, references, observations, profiles


def match_return(row, durable):
    keys = (
        "sample_id",
        "raw_answer",
        "finish_reason",
        "stop_reason",
        "profile_sha256",
        "output_token_ids",
        "prompt_token_count",
    )
    if (
        any(durable.get(key) != row.get(key) for key in keys)
        or durable["prompt_ids_sha256"] != row["prompt_token_ids_sha256"]
    ):
        raise ValueError(f'Durable engine output differs for {row["job_id"]}')
    if "unexpected_prompt_token_ids" in durable:
        ids = durable["unexpected_prompt_token_ids"]
        if (
            data.digest(ids) != durable["prompt_ids_sha256"]
            or len(ids) != durable["prompt_token_count"]
        ):
            raise ValueError(f'Unexpected prompt IDs differ for {row["job_id"]}')


def verify_archive(path):
    """Verify ZIP contents, request identity, frozen provenance, and rescore every row."""
    with zipfile.ZipFile(path) as archive:
        checksums = json.loads(archive.read("SHA256SUMS.json"))
        required = set((*PROOF_FILES, *REPORT_FILES))
        members = set(checksums)
        if (
            not required.issubset(members)
            or any(
                not re.fullmatch(SESSION_PATTERN, name) for name in members - required
            )
            or sorted(archive.namelist()) != sorted(members | {"SHA256SUMS.json"})
        ):
            raise ValueError("Return archive members differ")
        for name, expected in checksums.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError(f"Archive SHA-256 mismatch for {name}")
        run = json.loads(archive.read("evidence/run.json"))
        config = json.loads(archive.read("evidence/config.json"))
        source = json.loads(archive.read("evidence/source-code.json"))
        if (
            run["config_sha256"] != checksums["evidence/config.json"]
            or run["input_manifest_sha256"]
            != checksums["evidence/input-manifest.jsonl"]
            or run["source_code_sha256"] != checksums["evidence/source-code.json"]
            or run["runtime_lock_sha256"] != checksums["evidence/runtime-lock.json"]
        ):
            raise ValueError(
                "Run identity differs from the archived config, source or inputs"
            )
        if (
            run["repo_id"] != data.REPO_ID
            or config["dataset"]
            != {"repo_id": run["repo_id"], "revision": run["revision"]}
            or not data.dataset_catalog.known_public_release(
                run["revision"], run["index_sha256"]
            )
        ):
            raise ValueError("Unexpected release identity")
        known_manifests = {
            (f"{scope}/{suffix}", values[index], values[0])
            for scope, values in [
                *data.dataset_catalog.SCOPES.items(),
                *data.dataset_catalog.RGB16_SCOPES.items(),
                *data.dataset_catalog.PAIRED_MANIFESTS.items(),
                *data.dataset_catalog.PREVIOUS_MANIFESTS.items(),
            ]
            for suffix, index in (("metadata.jsonl", 2), ("static/metadata.jsonl", 3))
        }
        overlay = data.paired_inputs.manifest_record()
        if not run["manifests"] or any(
            entry != overlay
            and (entry["path"], entry["sha256"], entry["rows"]) not in known_manifests
            for entry in run["manifests"]
        ):
            raise ValueError("Unexpected published manifest identity")
        if not source:
            raise ValueError("Source code identity is missing")
        for sha in source.values():
            data.require_sha(sha)
        with archive.open("evidence/input-manifest.jsonl") as stream:
            cases = data.read_jsonl(stream)
        scopes = {data.case_scope(case) for case in cases}
        if any(
            scope in scopes
            and (
                not overlay["revision"]
                or overlay not in run["manifests"]
                or data.paired_inputs.source_scope(scope) in scopes
            )
            for scope in data.paired_inputs.LEGACY_SCOPES
        ):
            raise ValueError(
                "Historical T5 case provenance or reference version differs"
            )
        by_job = {}
        static_keys = set()
        if (
            not run["models"]
            or not set(run["models"]).issubset(data.MODELS)
            or len(set(run["models"])) != len(run["models"])
            or run["logical_rows"] != sum(len(case["aliases"]) for case in cases)
        ):
            raise ValueError("Invalid model or logical trial count")
        for case in cases:
            validate_case(case)
            if case["modality"] == "static":
                key = (data.case_scope(case), case["model_input_sha256"])
                if key in static_keys:
                    raise ValueError(f"Duplicate static input {key}")
                static_keys.add(key)
            for model in run["models"]:
                key = job_id(case, model)
                if key in by_job:
                    raise ValueError(f"Duplicate job ID {key}")
                by_job[key] = (case, model)
        if len(cases) != run["canonical_cases"] or len(by_job) != run["planned_calls"]:
            raise ValueError("Planned case or call counts differ")
        with archive.open("results.jsonl") as stream:
            rows = data.read_jsonl(stream)
        returns, references, observations, profiles = read_session_proof(
            archive, members
        )
        verified_deep = set()
        used_references = set()
        execution = json.loads(archive.read("evidence/execution.json"))
        schedule = execution["schedule"]
        if schedule is not None:
            model_gpus = schedule["model_gpus"]
            assigned = {
                job_id(case, model): devices[index % len(devices)]
                for model, devices in model_gpus.items()
                for index, case in enumerate(cases)
            }
            if assigned.keys() != by_job.keys():
                raise ValueError("GPU allocation does not cover the plan")
        else:
            assigned = {}
        for row in rows:
            pair = by_job.pop(row["job_id"], None)
            if pair is None:
                raise ValueError(f'Unknown or duplicated result {row["job_id"]}')
            case, model = pair
            if row["state"] != "PENDING" and row["gpu_uuid"] != assigned.get(
                row["job_id"]
            ):
                raise ValueError(
                    f'Attempt differs from frozen GPU allocation for {row["job_id"]}'
                )
            durable = returns.pop(row["job_id"], None)
            if row["state"] == "RETURNED":
                receipt = row["input_receipt"]
                if durable:
                    session, gpu, recipe, record = durable
                    if gpu != row["gpu_uuid"] or recipe != row["recipe"]:
                        raise ValueError(
                            f'Durable return has the wrong GPU or recipe for {row["job_id"]}'
                        )
                    match_return(row, record)
                if row["technical_pass"] is True:
                    if (
                        not durable
                        or not isinstance(row["raw_answer"], str)
                        or not isinstance(receipt, dict)
                        or not row["prompt_token_count"]
                        or not isinstance(row["output_token_ids"], list)
                        or receipt["prompt_sha256"] != row["prompt_sha256"]
                        or receipt["prompt_ids_sha256"]
                        != row["prompt_token_ids_sha256"]
                        or receipt["reference_key"] != record["reference_key"]
                        or profiles.get(session) != row["profile_sha256"]
                    ):
                        raise ValueError(
                            f'Missing validation receipt for {row["job_id"]}'
                        )
                    proof_key = (
                        session,
                        receipt["reference_key"],
                        row["prompt_token_ids_sha256"],
                        row["prompt_token_count"],
                    )
                    used_references.add(proof_key)
                    if receipt["deep_observation"]:
                        observation = (
                            session,
                            receipt["reference_key"],
                            row["sample_id"],
                        )
                        if observation not in observations or not isinstance(
                            receipt["admission"], dict
                        ):
                            raise ValueError(
                                f'Missing representative GPU observation for {row["job_id"]}'
                            )
                        verified_deep.add(proof_key)
                if row["prompt_token_ids_sha256"] is not None:
                    data.require_sha(row["prompt_token_ids_sha256"])
                if row["prompt_token_ids"] is not None and (
                    data.digest(row["prompt_token_ids"])
                    != row["prompt_token_ids_sha256"]
                    or len(row["prompt_token_ids"]) != row["prompt_token_count"]
                ):
                    raise ValueError(f'Prompt token IDs differ for {row["job_id"]}')
            elif (
                durable is not None
                or row["raw_answer"] is not None
                or row["technical_pass"] is not None
                or row["input_receipt"] is not None
            ):
                raise ValueError(f'Unexpected returned evidence for {row["job_id"]}')
            attempt = dict(
                state=row["state"],
                answer=row,
                gpu_uuid=row["gpu_uuid"],
                error=row["error"],
                started_ns=row["started_ns"],
                finished_ns=row["finished_ns"],
            )
            expected = result_row(case, model, config, run["config_sha256"], attempt)
            if row != expected:
                raise ValueError(f'Rescoring differs for {row["job_id"]}')
        if by_job:
            raise ValueError(f"Missing {len(by_job)} planned result rows")
        if returns:
            raise ValueError(
                f"{len(returns)} durable engine returns are absent from the results"
            )
        if not used_references.issubset(references & verified_deep):
            raise ValueError(
                "Some calls lack a matching full CPU prompt and successful GPU observation"
            )
        summary = summarize(rows)
        if summary != json.loads(archive.read("summary.json")):
            raise ValueError("Rescored summary differs")
        rebuilt_csv = io.StringIO(newline="")
        csv_rows(rebuilt_csv, rows)
        if rebuilt_csv.getvalue().encode() != archive.read("results.csv"):
            raise ValueError("Rescored CSV differs")
    return {
        "status": "VERIFIED",
        "archive": str(path),
        "sha256": data.file_sha(path),
        **summary["overall"],
    }
