"""Frozen input identity checks reused by the subset builder."""
import hashlib
import json
from pathlib import Path
import re
from . import paired_inputs
from . import release as current_release

FIXED_STATIC_FRAMES = {"T1": (0, 1, 2, 3), "T4": (0, 179)}


def json_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid SHA-256: {value!r}")
    return value


def parameters(row):
    return row.get("parameters") or row.get("source_parameters") or {}


def image_size(image):
    if image["frame_index"] is not None:
        return (768, 512)
    size = tuple(image.get("size_wh", (160, 160)))
    if size not in ((160, 160), (768, 512)):
        raise ValueError("Unexpected independent reference geometry")
    return size


def static_frame_indices(parent):
    task = parent["task"]
    if task in FIXED_STATIC_FRAMES:
        return list(FIXED_STATIC_FRAMES[task])
    source = parameters(parent)
    if task == "T5":
        return sorted(event["onset_frame"] + 15 for event in source["events"])
    onset = source.get("onset_frame", source.get("onset"))
    if not isinstance(onset, int):
        raise ValueError(
            f'Static midpoint needs a source onset for {parent["sample_id"]}'
        )
    return [onset + 15]


def static_identity(scope, task, images, frames, prompt, choices, sample_id):
    """Check ordered stills and hash the exact model-visible input."""
    if any(type(index) is not int or not 0 <= index < 180 for index in frames):
        raise ValueError(f"Wrong ordered images for {sample_id}")
    indices = [image["frame_index"] for image in images]
    if task == "T5":
        roles = [image.get("role") for image in images]
        valid = (
            len(frames) in (1, 2)
            and roles == ["reference", *(["event_midpoint"] * len(frames))]
            and indices == [None, *frames]
            and frames == sorted(set(frames))
        )
    else:
        expected = FIXED_STATIC_FRAMES.get(task)
        valid = indices == frames and (
            frames == list(expected) if expected is not None else len(frames) == 1
        )
    if not valid:
        raise ValueError(f"Wrong ordered images for {sample_id}")
    for image in images:
        reference = image["frame_index"] is None
        image_scope = scope if reference else paired_inputs.source_scope(scope)
        current_release.payload_path(image_scope, image["path"])
        if reference:
            expected_size = (
                (768, 512) if paired_inputs.is_current_reference(scope) else (160, 160)
            )
            if image_size(image) != expected_size:
                raise ValueError(f"Wrong reference geometry for {sample_id}")
        require_sha(image["sha256"])
    pixels = [require_sha(image["pixel_sha256"]) for image in images]
    return digest([pixels, prompt, choices])
