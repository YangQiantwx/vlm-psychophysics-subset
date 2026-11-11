"""RGB16 T2/T3 states and lossless encoding, with independent pixel checks."""

import hashlib
import os
import shutil
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import av
import numpy as np
from . import raster, verify
from .records import atomic_json, file_sha


def rgb16(y):
    if not np.isfinite(y).all() or np.min(y) < 0 or np.max(y) > 1:
        raise ValueError("Invalid/clipped source luminance")
    q = np.rint(raster.srgb(y) * 65535).astype(np.uint16)
    return np.repeat(q[..., None], 3, axis=-1)


@lru_cache(maxsize=4)
def disk(area):
    return raster.disk_coverage(area=area)


def make_state(row):
    p = row["source_parameters"]
    shape = (p["height"], p["width"])
    if shape != (512, 768):
        raise ValueError("This protocol fixes source dimensions at768x512")
    if row["task"] == "T2":
        if p["centers_xy"] != [[224, 302], [544, 302]]:
            raise ValueError("Unexpected disk center")
        left, circle_qa = disk(p["disk_alpha_area_px2"])
        if abs(circle_qa["diameter_px"] - p["disk_diameter_px"]) > 1e-9:
            raise ValueError("Declared disk area/diameter disagree")
        right = np.roll(left, 320, axis=1)
        base_y = p["background_Y"] + (left + right) * (
            p["initial_dot_Y"] - p["background_Y"]
        )
        target = left if p["target_side"] == "LEFT" else right
        baseline, event = rgb16(base_y), rgb16(base_y + target * p["delta_Y"])
        qa = verify.source_state(row, baseline, event, target)
        qa["independent_circle_validation"] = circle_qa
        return baseline, event, qa
    angles = [
        0.0 if side == p["reference_side"] else p["angle_deg"] * p["tilt_sign"]
        for side in ("LEFT", "RIGHT")
    ]
    alpha = np.zeros(shape, np.float64)
    independent = np.zeros(shape, np.float64)
    for center, angle in zip(p["centers_xy"], angles):
        kw = {
            "shape": shape,
            "center": tuple(center),
            "width": p["bar_wh"][0],
            "height": p["bar_wh"][1],
        }
        alpha += raster.strip_coverage(angle, **kw)
        independent += raster.geos_coverage(angle, **kw)
    diff = float(np.max(abs(alpha - independent)))
    if diff >= 1e-9 or np.max(alpha) > 1 + 1e-12:
        raise ValueError(("Independent coverage or overlap failure", diff))
    baseline = rgb16(np.full(shape, p["background_Y"]))
    event = rgb16(p["background_Y"] + (p["foreground_Y"] - p["background_Y"]) * alpha)
    qa = verify.source_state(row, baseline, event, alpha)
    qa["independent_GEOS_max_coverage_difference"] = diff
    return baseline, event, qa


def validate_video(row, path, base, event):
    """Existing frame-by-frame RGB16/source equality check, without model arrays."""
    p = row["source_parameters"]
    on, off, n = p["onset_frame"], p["offset_frame"], p["frames"]
    actual_hash = hashlib.sha256()
    count, pts_error = 0, 0.0
    with av.open(str(path)) as container:
        if container.streams.video[0].codec_context.format.name != "gbrp16le":
            raise ValueError("Encoder dropped channel precision")
        for count, frame in enumerate(container.decode(video=0), 1):
            i = count - 1
            actual = frame.to_ndarray(format="rgb48le")
            expected = event if on <= i < off else base
            if actual.dtype != np.uint16 or not np.array_equal(actual, expected):
                raise ValueError(
                    ("Lossless frame/sequence mismatch", row["sample_id"], i)
                )
            actual_hash.update(actual.tobytes())
            pts_error = max(
                pts_error, abs(float(frame.pts * frame.time_base) - i / p["fps"])
            )
    if count != n or pts_error > 0.001:
        raise ValueError(("Encoded timing/stream mismatch", row["sample_id"]))
    return {
        "stream_sha256": actual_hash.hexdigest(),
        "frames": count,
        "max_PTS_error_s": pts_error,
        "max_decoded_code_error": 0,
    }


def _encode_one(task):
    row, state, bank_path, run_dir, minimum_free_bytes = task
    run_dir = Path(run_dir)
    if shutil.disk_usage(run_dir).free < minimum_free_bytes:
        raise RuntimeError(
            "Disk reserve reached; completed ready records are preserved"
        )
    bank = np.load(bank_path, mmap_mode="r")
    base, event = [bank[i] for i in state["indices"]]
    p = row["source_parameters"]
    video = run_dir / "videos" / (row["sample_id"] + ".mkv")
    partial = video.with_name(video.name + ".partial")
    video.parent.mkdir(exist_ok=True)
    on, off, n = p["onset_frame"], p["offset_frame"], p["frames"]
    expected_hash = hashlib.sha256()
    with av.open(str(partial), "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=p["fps"])
        stream.width, stream.height = p["width"], p["height"]
        stream.pix_fmt = "gbrp16le"
        stream.codec_context.thread_count = 1
        stream.codec_context.colorspace = 0
        stream.options = {"level": "3", "coder": "1", "context": "1", "slicecrc": "1"}
        for i in range(n):
            rgb = event if on <= i < off else base
            expected_hash.update(rgb.tobytes())
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb48le")
            frame.pts, frame.time_base = i, Fraction(1, p["fps"])
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    qa = validate_video(row, partial, base, event)
    if qa["stream_sha256"] != expected_hash.hexdigest():
        raise ValueError(("Encoded timing/stream mismatch", row["sample_id"]))
    with partial.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(partial, video)
    ready = {
        "sample_id": row["sample_id"],
        "video_relative": str(video.relative_to(run_dir)),
        "video_sha256": file_sha(video),
        "stream_sha256": qa["stream_sha256"],
        "source_parameters": p,
        "frames": qa["frames"],
        "master_bits_per_channel": 16,
        "max_decoded_code_error": 0,
        "max_PTS_error_s": qa["max_PTS_error_s"],
    }
    # A small sidecar closes the encoded-video -> ready append recovery window.
    atomic_json(video.with_suffix(".ready.json"), ready)
    return ready
