"""Request and response formats for hosted video and frame-list adapters."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
from io import BytesIO
import json
import os
import re
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

from ..catalog import require

GEMINI = "https://generativelanguage.googleapis.com/v1beta"
QWEN = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
ASTRA = "https://api.openai.com/v1"
CLAUDE = "https://api.anthropic.com/v1"
KIMI = "https://api.moonshot.ai/v1"
CLAUDE_API_VERSION = "2026-10-21"
CLAUDE_INLINE_LIMIT = 128_000_000
CLAUDE_FILE_EXPIRY_SECONDS = 7 * 24 * 60 * 60
CLAUDE_MODELS = {"claude-opus-5-5", "claude-fable-5-1"}
CLAUDE_IMAGE_TRANSFORMATIONS = {"oversized_image": "error"}
GEMINI_MODELS = {"gemini-4-carbon", "gemini-4-argon", "gemini-4-hydrogen", "gemini-3.8-flash", "gemini-3.1-pro-preview"}
QWEN_ENDPOINT = re.compile(
    r"https://(?:(?:cn-hongkong\.)?dashscope(?:-intl|-us)?\.aliyuncs\.com|"
    r"[a-z0-9-]+\.[a-z0-9-]+\.maas\.aliyuncs\.com)/compatible-mode/v1"
)
QWEN_FRAME_MODEL = re.compile(
    r"qwen3\.[5-8]-(?:max|plus|flash|[0-9][A-Za-z0-9.-]*b)(?:-[0-9-]+)?"
)
HOSTED = {
    "gemini": ("GEMINI_API_KEY", 24, {"video/mp4"}),
    "qwen": ("DASHSCOPE_API_KEY", 10, {"video/mp4", "video/x-matroska"}),
    "qwen-frames": ("DASHSCOPE_API_KEY", 10, {"video/mp4"}),
    "astra": ("OPENAI_API_KEY", None, {"video/mp4"}),
    "claude": ("ANTHROPIC_API_KEY", None, {"video/mp4"}),
    "kimi": ("MOONSHOT_API_KEY", None, {"video/mp4"}),
}
ORDERED_IMAGES = {"astra", "claude"}
FRAME_LISTS = {"qwen-frames", *ORDERED_IMAGES}
FULL_FRAME_FIELDS = {
    "media_io_kwargs": {"video": {"num_frames": -1, "fps": -1}},
    "mm_processor_kwargs": {"do_sample_frames": False},
}
SOURCE_FRAMES, SOURCE_FPS = 180, 60
SOURCE_WIDTH, SOURCE_HEIGHT = 768, 512
PNG_COMPRESSION = {"qwen-frames": 6, "astra": 1, "claude": 6, "nvidia-internal": 6}


def model_provider(model):
    if re.fullmatch(r"gpt-6-astra(?:-[0-9]{4}-[0-9]{2}-[0-9]{2})?", model):
        return "astra"
    if model in CLAUDE_MODELS:
        return "claude"
    if model in GEMINI_MODELS:
        return "gemini"
    if QWEN_FRAME_MODEL.fullmatch(model):
        return "qwen-frames"
    return None


def fixed_endpoint(provider):
    return {"gemini": GEMINI, "astra": ASTRA, "claude": CLAUDE, "kimi": KIMI}.get(provider)


def vllm_endpoint(base):
    url = urlsplit(base)
    local = url.hostname in ("localhost", "127.0.0.1", "::1")
    require(
        (url.scheme == "https" or (local and url.scheme == "http"))
        and url.hostname
        and url.path.endswith("/v1")
        and ".." not in url.path and "//" not in url.path
        and not any((url.username, url.password, url.query, url.fragment)),
        "Use an HTTPS API base URL ending in /v1, or a localhost tunnel",
    )
    return base


def endpoint(provider):
    if base := fixed_endpoint(provider):
        return base
    base = os.environ.get("DASHSCOPE_BASE_URL", QWEN)
    require(
        bool(QWEN_ENDPOINT.fullmatch(base)),
        "Use an official DashScope compatible-mode HTTPS endpoint",
    )
    return base


def requested_fps(provider, model):
    if provider == "qwen-frames" and model.startswith("qwen3.8-"):
        # DashScope does not document frame-list timestamps for this series.
        return None
    return HOSTED[provider][1]


def validate_config(config):
    provider = config["provider"]
    model = config["requested_model"]
    if provider == "gemini":
        require(
            bool(re.fullmatch(r"gemini-[A-Za-z0-9_.-]+", model)),
            "Use an exact Gemini model ID",
        )
    if provider == "astra":
        require(model.startswith("gpt-6-astra"), "Use a GPT-6 Astra model ID")
    if provider == "kimi":
        require(model == 'kimi-k3.1-pro', 'Use the Kimi K3.1 Pro model ID')
    if provider == "claude":
        require(model in CLAUDE_MODELS, "Use claude-opus-5-5 or claude-fable-5-1")
        require(
            config.get("api_version") == CLAUDE_API_VERSION,
            "Claude Messages API version differs",
        )
    if provider == "qwen-frames":
        require(
            bool(QWEN_FRAME_MODEL.fullmatch(model)),
            "Use an exact supported Qwen3.5–3.8 vision model ID, e.g. qwen3.8-max-0902",
        )
    base = config["endpoint"]
    fixed = fixed_endpoint(provider)
    require(
        base == fixed if fixed else bool(QWEN_ENDPOINT.fullmatch(base)),
        "Hosted API endpoint differs",
    )
    # Retain verification of older Qwen frame-list archives that requested 10 fps.
    allowed_fps = (requested_fps(provider, model), HOSTED[provider][1])
    require(
        config["video_fps"] in allowed_fps,
        "Hosted video sampling setting differs",
    )


def ordered_image_settings(provider, video_fps=None):
    settings = {
        "delivery": "ordered_lossless_png",
        "frames": SOURCE_FRAMES,
        "source_fps": SOURCE_FPS,
        "requested_fps": video_fps,
        "png_compression": PNG_COMPRESSION[provider],
    }
    if provider == "claude":
        settings["image_transformations"] = CLAUDE_IMAGE_TRANSFORMATIONS
    return settings


def frame_settings(provider, video_fps=None):
    if provider in FRAME_LISTS:
        return ordered_image_settings(provider, video_fps)
    return FULL_FRAME_FIELDS


def ordered_pngs(payload, expected_rgb_sha=None, *, compression=6):
    """Decode each original frame once and preserve RGB values in the PNG list."""
    import av
    from PIL import Image

    images = []
    digest = hashlib.sha256()
    with av.open(BytesIO(payload)) as container:
        require(len(container.streams.video) == 1, "Expected exactly one MP4 video stream")
        (stream,) = container.streams.video
        stream.codec_context.thread_count = 1
        require(stream.average_rate == SOURCE_FPS, "Expected a 60 fps video")
        for i, frame in enumerate(container.decode(stream)):
            require(
                (frame.width, frame.height) == (SOURCE_WIDTH, SOURCE_HEIGHT)
                and i < SOURCE_FRAMES,
                "Unexpected frame size or count",
            )
            require(
                frame.pts is not None and frame.time_base is not None
                and frame.pts * frame.time_base == Fraction(i, SOURCE_FPS),
                "Expected timestamps i/60 for all 180 frames in order",
            )
            rgb = frame.to_ndarray(format="rgb24")
            digest.update(rgb)
            png = BytesIO()
            Image.fromarray(rgb).save(png, format="PNG", compress_level=compression)
            images.append("data:image/png;base64," + base64.b64encode(png.getvalue()).decode("ascii"))
    require(len(images) == SOURCE_FRAMES, "Expected all 180 frames")
    rgb_sha = digest.hexdigest()
    require(
        expected_rgb_sha is None or rgb_sha == expected_rgb_sha,
        "Frame-list RGB differs from the verified evaluation reader output",
    )
    return images, rgb_sha


def frame_list_request(provider, model, prompt, parts, max_tokens, generation):
    if provider not in ORDERED_IMAGES:
        return {
            "model": model,
            "stream": False,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": parts + [{"type": "text", "text": prompt}]}],
            **generation,
        }
    images = [
        url
        for part in parts
        for url in (part["video"] if part["type"] == "video" else [part["image_url"]["url"]])
    ]
    if provider == "astra":
        content = [
            {"type": "input_image", "image_url": url, "detail": "original"}
            for url in images
        ]
        return {
            "model": model,
            "input": [{"role": "user", "content": content + [{"type": "input_text", "text": prompt}]}],
            "max_output_tokens": max_tokens,
            "reasoning": generation["reasoning"],
            "tools": [],
            "tool_choice": "none",
            "store": False,
            "stream": False,
            "truncation": "disabled",
        }
    content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": url.split(",", 1)[1],
            },
            "transformations": CLAUDE_IMAGE_TRANSFORMATIONS,
        }
        for url in images
    ]
    return {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content + [{"type": "text", "text": prompt}]}],
        "stream": False,
        **generation,
    }


def claude_headers(credential):
    headers = {"Authorization": "Bearer " + credential, "anthropic-version": CLAUDE_API_VERSION}
    if workspace := os.environ.get("ANTHROPIC_WORKSPACE_ID"):
        headers["anthropic-workspace-id"] = workspace
    return headers


def upload_claude_png(data, digest, credential):
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    boundary = uuid4().hex
    payload = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{digest}.png"\r\n'
        f'Content-Type: image/png\r\n\r\n'
    ).encode() + data + (
        f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="expires_in_seconds"\r\n\r\n'
        f'{CLAUDE_FILE_EXPIRY_SECONDS}\r\n--{boundary}--\r\n'
    ).encode()
    request = Request(CLAUDE + "/files", data=payload, method="POST", headers={
        **claude_headers(credential), "Content-Type": f"multipart/form-data; boundary={boundary}",
    })
    opener = build_opener(NoRedirect)
    for attempt in range(3):
        try:
            with opener.open(request, timeout=120) as response:
                record = json.load(response)
            break
        except HTTPError as error:
            with error:
                status = error.code
                delay = error.headers.get("Retry-After", "5")
            if status != 429 or attempt == 2:
                raise ValueError(f"Anthropic PNG upload failed (HTTP {status}).") from None
            time.sleep(min(60, int(delay)) if delay.isdecimal() else 5)
    file_id = record.get("id")
    require(
        isinstance(file_id, str) and bool(re.fullmatch(r"file_[A-Za-z0-9_-]+", file_id))
        and record.get("size_bytes") == len(data) and record.get("mime_type") == "image/png"
        and bool(record.get("expires_at")),
        "Anthropic Files response did not confirm the PNG upload and expiry",
    )
    return file_id


def claude_file_parts(body, credential):
    """Reference the same PNG bytes in their original positions, uploading duplicates once per trial."""
    parts = body["messages"][0]["content"]
    images = [part for part in parts if part["type"] == "image"]
    ordered = [part["source"]["data"] for part in images]

    def upload(encoded):
        data = base64.b64decode(encoded, validate=True)
        return upload_claude_png(data, hashlib.sha256(data).hexdigest(), credential)

    unique = dict.fromkeys(ordered)
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = dict(zip(unique, pool.map(upload, unique)))
    for image, encoded in zip(images, ordered):
        image["source"] = {"type": "file", "file_id": files[encoded]}
    return len(images)


def upload_qwen_media(data, mime, config, credential):
    """Use DashScope's model/account-bound 48-hour storage; never expose labeled filenames."""
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    base = config['endpoint'].replace('/compatible-mode/v1', '/api/v1')
    req = Request(base + '/uploads?action=getPolicy&model=' + config['requested_model'],
                  headers={'Authorization': 'Bearer ' + credential, 'Content-Type': 'application/json'})
    opener = build_opener(NoRedirect)
    try:
        with opener.open(req, timeout=60) as response:
            policy = json.load(response)['data']
        host = urlsplit(policy['upload_host'])
        require(host.scheme == 'https' and host.hostname and host.hostname.endswith('.aliyuncs.com')
                and not host.username and not host.password, 'Unexpected DashScope upload host')
        require(len(data) <= float(policy['max_file_size_mb']) * 1_000_000,
                'Media exceeds the model temporary-upload limit')
        filename = hashlib.sha256(data).hexdigest() + ('.png' if mime == 'image/png' else '.mp4')
        key = policy['upload_dir'].rstrip('/') + '/' + uuid4().hex + '-' + filename
        fields = {'OSSAccessKeyId': policy['oss_access_key_id'], 'Signature': policy['signature'],
                  'policy': policy['policy'], 'x-oss-object-acl': policy['x_oss_object_acl'],
                  'x-oss-forbid-overwrite': policy['x_oss_forbid_overwrite'], 'key': key,
                  'success_action_status': '200'}
        boundary = uuid4().hex
        payload = b''.join((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                            f'{value}\r\n').encode() for name, value in fields.items())
        payload += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                    f'Content-Type: {mime}\r\n\r\n').encode() + data + f'\r\n--{boundary}--\r\n'.encode()
        req = Request(policy['upload_host'], data=payload, method='POST',
                      headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})
        with opener.open(req, timeout=180) as response:
            require(response.status == 200, 'DashScope temporary upload failed')
        return 'oss://' + key
    except HTTPError as error:
        with error:
            status = error.code
        raise ValueError(f'DashScope temporary upload failed (HTTP {status}).') from None


def qwen_file_parts(body, config, credential):
    """Replace inline bytes with temporary references without changing part/frame order."""
    parts = body['messages'][0]['content']
    urls = []
    for part in parts:
        if part['type'] == 'video': urls.extend(part['video'])
        elif part['type'] in {'video_url', 'image_url'}: urls.append(part[part['type']]['url'])
    unique = dict.fromkeys(urls)

    def upload(url):
        header, encoded = url.split(',', 1)
        mime = header.removeprefix('data:').removesuffix(';base64')
        require(mime in {'image/png', 'video/mp4'}, 'Unsupported temporary media MIME')
        return upload_qwen_media(base64.b64decode(encoded, validate=True), mime, config, credential)

    with ThreadPoolExecutor(max_workers=4) as pool:
        mapping = dict(zip(unique, pool.map(upload, unique)))
    for part in parts:
        if part['type'] == 'video': part['video'] = [mapping[url] for url in part['video']]
        elif part['type'] in {'video_url', 'image_url'}:
            entry = part[part['type']]
            entry['url'] = mapping[entry['url']]
    return len(urls)


def astra_completion(document):
    output = document["output"]
    parts = [p for item in output if item["type"] == "message" for p in item["content"]]
    content = "".join(p["text"] for p in parts if p["type"] == "output_text")
    status = document["status"]
    incomplete = (document.get("incomplete_details") or {}).get("reason")
    stop = "stop" if status == "completed" else incomplete or status
    if status == "incomplete" and incomplete == "max_output_tokens":
        stop = "length"
    if status == "completed" and any(p["type"] != "output_text" for p in parts):
        stop = "refusal_or_nontext"
    return {
        "model": document["model"],
        "id": document.get("id"),
        "usage": document.get("usage"),
        "choices": [{
            "finish_reason": stop,
            "message": {
                "content": content,
                "tool_calls": any(item["type"] not in {"message", "reasoning"} for item in output),
            },
        }],
    }


def validate_payload_size(provider, base, size):
    if provider == "qwen-frames":
        hostname = urlsplit(base or "").hostname or ""
        workspace = hostname.endswith(".maas.aliyuncs.com")
        limit = 64_000_000 if workspace else 16 * 2**20
        hint = "" if workspace else "; set DASHSCOPE_BASE_URL to the official workspace endpoint"
    elif provider == "astra":
        limit, hint = 512_000_000, ""
    elif provider == "claude":
        limit, hint = CLAUDE_INLINE_LIMIT, ""
    elif provider == "gemini":
        limit, hint = 20_000_000, ""
    else:
        return
    require(size <= limit, f"{provider} request is {size} bytes, above the {limit}-byte limit{hint}")


def claude_completion(document):
    parts = document["content"]
    stop = document["stop_reason"]
    finish = {
        "end_turn": "stop",
        "max_tokens": "length",
        "model_context_window_exceeded": "length",
    }.get(stop, stop)
    if (document.get("stop_details") or {}).get("type") == "refusal":
        finish = "refusal"
    return {
        "model": document["model"],
        "id": document.get("id"),
        "usage": document.get("usage"),
        "choices": [{
            "finish_reason": finish,
            "message": {
                "content": "".join(part["text"] for part in parts if part["type"] == "text"),
                "tool_calls": any(
                    part["type"] not in {"text", "thinking", "redacted_thinking"}
                    for part in parts
                ),
            },
        }],
    }


def request(provider, model, prompt, media, max_tokens, fps, *, panel=False, generation=None):
    require(bool(media) and all(m in (HOSTED[provider][2] | ({'image/png'} if panel else set()))
                               for m, _ in media), "Unsupported hosted video MIME type")
    require(panel or len(media) == 1, 'Expected one video')
    require(sum(m.startswith('video/') for m, _ in media) <= 1, 'Expected at most one video')
    if provider == "gemini":
        parts = []
        for mime, data in media:
            part = {"inlineData": {"mimeType": mime, "data": data}}
            if mime.startswith('video/'):
                part['videoMetadata'] = {'fps': fps}
                if panel: part['mediaProcessing'] = 'STATIC'
            parts.append(part)
        return {
            "contents": [{"role": "user", "parts": parts + [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, **(generation or {})},
        }
    parts = []
    for mime, data in media:
        url = f"data:{mime};base64,{data}"
        if provider == 'qwen' and not panel:
            require(len(url) < 10_000_000, "DashScope inline video exceeds 10 MB encoded")
        kind = 'image_url' if mime == 'image/png' else 'video_url'
        part = {'type': kind, kind: {'url': url}}
        if provider == 'qwen':
            if kind == 'video_url': part['fps'] = fps
            if panel: part.update(min_pixels=SOURCE_WIDTH * SOURCE_HEIGHT, max_pixels=SOURCE_WIDTH * SOURCE_HEIGHT)
        parts.append(part)
    return {
        "model": model,
        "stream": False,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": parts + [{"type": "text", "text": prompt}]}
        ],
        **(generation or {}),
    }


def gemini_completion(document):
    """Project the native envelope onto the fields used by the common scorer."""
    (candidate,) = document["candidates"]
    parts = candidate["content"]["parts"]
    text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
    reason = candidate.get("finishReason")
    finish = {"STOP": "stop", "MAX_TOKENS": "length"}.get(reason, reason)
    return {
        "model": document["modelVersion"],
        "id": document.get("responseId"),
        "usage": document.get("usageMetadata"),
        "choices": [
            {
                "finish_reason": finish,
                "message": {
                    "content": text,
                    "tool_calls": any("functionCall" in part for part in parts),
                },
            }
        ],
    }
