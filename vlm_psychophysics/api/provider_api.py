"""Single-request, no-tool API evaluation; dry-run unless --execute is given.

Inputs: JSON list of {sample_id, prompt, video_path, video_sha256}.
Credentials are read from the selected environment variable and never persisted.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse

ROUTES = {
    "nvidia-internal": ("https://inference-api.nvidia.com/v1", "NVIDIA_API_KEY", None),
    "nvidia-public": (
        "https://integrate.api.nvidia.com/v1",
        "NVIDIA_API_KEY",
        "moonshotai/kimi-k2.5",
    ),
    "moonshot": ("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY", "kimi-k3.1-pro"),
    "bigmodel": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY", "glm-4.6v"),
}


def validate(row: dict, root: Path) -> Path:
    assert row["sample_id"] and row["prompt"]
    assert (
        row.get("task") != "T5"
    ), "T5 reference-image plus video admission is not yet qualified for these provider routes."
    path = root / (row.get("video_path") or row["path"])
    assert path.is_file(), path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == row["video_sha256"], f'Source SHA mismatch: {row["sample_id"]}'
    return path


def parameters(
    provider: str, model: str, prompt: str, video_url: str, max_tokens: int
) -> dict:
    # No tools are offered and there is no tool executor or agent loop.
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "stream": False,
    }
    body["max_completion_tokens" if provider == "nvidia-internal" else "max_tokens"] = (
        max_tokens
    )
    if provider == "bigmodel":
        thinking = "enabled" if model.startswith("glm-5.3-flash") else "disabled"
        body["extra_body"] = {"thinking": {"type": thinking}}
    if provider == "nvidia-public":
        body["extra_body"] = {"chat_template_kwargs": {"thinking": False}}
    return body


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", choices=ROUTES, required=True)
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--key-env")
    ap.add_argument(
        "--inventory", action="store_true", help="Read account /models; no inference."
    )
    ap.add_argument("--inputs", type=Path)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument(
        "--video-url",
        help="BigModel HTTPS object URL with the source SHA as its filename.",
    )
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    base_url, key_env, default_model = ROUTES[args.provider]
    base_url = args.base_url or base_url
    key_env = args.key_env or key_env
    model = args.model or default_model
    args.out.mkdir(parents=True, exist_ok=True)
    if args.inventory:
        from openai import OpenAI

        client = OpenAI(
            base_url=base_url, api_key=os.environ[key_env], max_retries=0, timeout=120
        )
        result = client.models.list().model_dump(mode="json")
        (args.out / "account-models.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(
            json.dumps({"status": "ACCOUNT_MODELS_READ", "count": len(result["data"])})
        )
        return
    assert (
        model
    ), "Internal gateway requires an exact model ID from account-models.json."
    assert args.inputs and args.limit > 0
    if args.inputs.suffix == ".jsonl":
        with args.inputs.open() as stream:
            rows = [
                json.loads(line)
                for line in islice(
                    (line for line in stream if line.strip()), args.limit
                )
            ]
    else:
        document = json.loads(args.inputs.read_text())
        rows = (document["example_inputs"] if isinstance(document, dict) else document)[
            : args.limit
        ]
    assert rows and len({r["sample_id"] for r in rows}) == len(rows)
    paths = [validate(r, args.root) for r in rows]
    if args.provider == "moonshot" and any(
        path.suffix.lower() != ".mp4" for path in paths
    ):
        ap.error("Select a published MP4 for the Moonshot video upload")
    if args.provider == "bigmodel":
        if len(rows) != 1 or not args.video_url:
            ap.error("BigModel takes one SHA-named HTTPS video URL per invocation")
        url = urlparse(args.video_url)
        name = rows[0]["video_sha256"] + paths[0].suffix
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.fragment
            or url.path not in (f"/{name}", f"/video/{name}", f"/videos/{name}")
        ):
            ap.error("Use a neutral host and a SHA-named object path for BigModel")
    preview = {
        "status": "PREPARED_NO_MODEL_CALLS",
        "provider": args.provider,
        "endpoint": base_url,
        "model": model,
        "tool_executor": False,
        "sampling_policy": "provider defaults; seed/temperature not imposed",
        "closed_provider_input_parity": "UNVERIFIED",
        "items": [
            {
                "sample_id": r["sample_id"],
                "video_sha256": r["video_sha256"],
                "video_bytes": p.stat().st_size,
                "request": parameters(
                    args.provider,
                    model,
                    r["prompt"],
                    "<validated-video-media>",
                    args.max_tokens,
                ),
            }
            for r, p in zip(rows, paths)
        ],
    }
    (args.out / "preflight.json").write_text(json.dumps(preview, indent=2) + "\n")
    if not args.execute:
        print(json.dumps({"status": "PREPARED_NO_MODEL_CALLS", "items": len(rows)}))
        return
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url, api_key=os.environ[key_env], max_retries=0, timeout=300
    )
    for row, path in zip(rows, paths):
        target = args.out / row["sample_id"]
        target.mkdir(exist_ok=False)  # A rerun never overwrites an original response.
        receipt = {
            "sample_id": row["sample_id"],
            "source_sha256": row["video_sha256"],
            "provider": args.provider,
            "endpoint": base_url,
            "requested_model": model,
            "started_unix": time.time(),
            "tool_executor": False,
        }
        (target / "request.json").write_text(
            json.dumps(
                {
                    **receipt,
                    "prompt": row["prompt"],
                    "max_tokens": args.max_tokens,
                    "media_mode": "video_url",
                },
                indent=2,
            )
            + "\n"
        )
        try:
            if args.provider == "moonshot":
                with path.open("rb") as stream:
                    uploaded = client.files.create(
                        file=(row["video_sha256"] + ".mp4", stream, "video/mp4"),
                        purpose="video",
                    )
                receipt["provider_file_id"] = uploaded.id
                media_url = f"ms://{uploaded.id}"
                (target / "upload.json").write_text(
                    uploaded.model_dump_json(indent=2) + "\n"
                )
            elif args.provider == "bigmodel":
                media_url = args.video_url
                # Admit the actual remote bytes, not merely a local namesake.
                import requests

                remote_sha, remote_bytes = hashlib.sha256(), 0
                with requests.get(
                    media_url, stream=True, timeout=60, allow_redirects=False
                ) as remote:
                    remote.raise_for_status()
                    assert remote.status_code == 200
                    for chunk in remote.iter_content(1 << 20):
                        remote_bytes += len(chunk)
                        assert remote_bytes <= path.stat().st_size
                        remote_sha.update(chunk)
                assert (
                    remote_bytes == path.stat().st_size
                    and remote_sha.hexdigest() == row["video_sha256"]
                )
                receipt["remote_media_source_sha256_verified"] = remote_sha.hexdigest()
            else:
                mime = {".mp4": "video/mp4", ".mkv": "video/x-matroska"}[
                    path.suffix.lower()
                ]
                media_url = (
                    f"data:{mime};base64,"
                    + base64.b64encode(path.read_bytes()).decode()
                )
            body = parameters(
                args.provider, model, row["prompt"], media_url, args.max_tokens
            )
            response = client.chat.completions.create(**body)
            (target / "raw-response.json").write_text(
                response.model_dump_json(indent=2) + "\n"
            )
            message = response.choices[0].message
            label = (message.content or "").strip().upper().rstrip(".!")
            valid = label in row.get("choices", ["LEFT", "RIGHT"])
            finish_reason = response.choices[0].finish_reason
            status = (
                "TRUNCATED"
                if finish_reason == "length"
                else (
                    "INCOMPLETE"
                    if finish_reason != "stop"
                    else (
                        "COMPLETE"
                        if valid and not message.tool_calls
                        else "FORMAT_OR_TOOL_CALL_FAILURE"
                    )
                )
            )
            receipt.update(
                status=status,
                answer=label if valid else None,
                returned_model=response.model,
                finish_reason=response.choices[0].finish_reason,
                tool_calls_returned=bool(message.tool_calls),
                usage=(
                    response.usage.model_dump(mode="json") if response.usage else None
                ),
            )
        except Exception as exc:
            # Provider errors remain terminal records; no implicit paid retries.
            receipt.update(
                status="PROVIDER_ERROR",
                error_type=type(exc).__name__,
                error_status=getattr(exc, "status_code", None),
            )
        receipt["finished_unix"] = time.time()
        (target / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(
            json.dumps({"sample_id": row["sample_id"], "status": receipt["status"]}),
            flush=True,
        )


if __name__ == "__main__":
    main()
