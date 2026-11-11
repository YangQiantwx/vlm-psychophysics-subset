"""Frozen Natural T4 motion geometry, RGB8 compositing and encoding."""

from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import av
import numpy as np
from scipy.ndimage import map_coordinates

W, H, F, FPS = 768, 512, 180, 60
MODES = ("same", "distinct")
COLOR = {"color_range": 2, "colorspace": 0, "color_primaries": 1, "color_trc": 13}
XY = np.indices((58, 58), dtype=np.float64)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, allow_nan=False)
        f.write("\n")


def cases(plan):
    for b in plan["blocks"]:
        for mode in MODES:
            for t in b["trials"]:
                sid = f"natural-t4-strict-v1-n{b['N']}-{mode}-l{b['layout']:02d}-c{b['level']:02d}-{t['truth'].lower()}"
                yield (sid, b, mode, t)


def sha_pixels(array):
    return hashlib.sha256(np.ascontiguousarray(array)).hexdigest()


def rgb8(values):
    assert (
        np.isfinite(values).all()
        and values.min() >= -1e-09
        and (values.max() <= 1 + 1e-09)
    )
    x = np.clip(values, 0, 1)
    return np.rint(
        255 * np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    ).astype(np.uint8)


@lru_cache(maxsize=4)
def background(path, expected):
    assert sha(path) == expected
    a = np.load(path, allow_pickle=False)
    assert a.shape == (H, W, 3)
    return (a, rgb8(a))


@lru_cache(maxsize=100)
def object_patch(path, expected):
    assert sha(path) == expected
    a = np.load(path, allow_pickle=False)
    assert (
        a.shape == (56, 56, 4)
        and np.isfinite(a).all()
        and (a.min() >= -1e-10)
        and (a.max() <= 1 + 1e-10)
    )
    assert (a[..., :3] <= a[..., 3, None] + 1e-10).all()
    y, x = np.where(a[..., 3] > 0)
    assert np.max(np.hypot(x + 0.5 - 28, y + 0.5 - 28)) + np.sqrt(2) < 28
    return np.moveaxis(a, 2, 0).copy()


def geometry(t, n, level):
    start = np.asarray(t["start"], dtype=np.float64)
    vel = np.asarray(t["velocity"], dtype=np.float64)
    assert start.shape == vel.shape == (n, 2)
    signal = t["signal_indices"]
    assert len(signal) == len(set(signal)) == n * level / 8
    assert np.allclose(
        vel[signal], [-32, 0] if t["truth"] == "LEFT" else [32, 0], rtol=0, atol=1e-12
    )
    assert np.allclose(np.linalg.norm(vel, axis=1), 32, rtol=0, atol=1e-09)
    stop = start + vel * (179 / 60)
    edges = np.minimum(np.minimum(start, stop), [W, H] - np.maximum(start, stop)) - 28
    closest = float("inf")
    for i in range(n):
        for j in range(i):
            p = start[i] - start[j]
            v = vel[i] - vel[j]
            speed = v @ v
            tmin = 0 if speed == 0 else np.clip(-(p @ v) / speed, 0, 179 / 60)
            closest = min(closest, float(np.linalg.norm(p + tmin * v)))
    assert edges.min() >= 1.001 - 1e-08 and closest >= 56.25 - 1e-07
    return (start, vel, float(edges.min()), closest)


def frame_at(points, patches, bg, base):
    image = base.copy()
    used = np.zeros((H, W), bool)
    contrast_counts = []
    for (cx, cy), patch in zip(points, patches, strict=True):
        left, top = (cx - 28, cy - 28)
        x, y = (int(np.floor(left)), int(np.floor(top)))
        assert x >= 0 and y >= 0
        h, w = (min(58, H - y), min(58, W - x))
        yy, xx = XY[:, :h, :w]
        coords = [yy - (top - y), xx - (left - x)]
        alpha = map_coordinates(
            patch[3], coords, order=1, mode="grid-constant", cval=0, prefilter=False
        )
        active = alpha > 0
        assert active.any()
        assert (
            np.max((xx[active] + x + 0.5 - cx) ** 2 + (yy[active] + y + 0.5 - cy) ** 2)
            <= 28**2 + 1e-08
        )
        support = used[y : y + h, x : x + w]
        assert not np.any(support & active)
        support[active] = True
        a = alpha[active, None]
        rgb = np.stack(
            [
                map_coordinates(
                    patch[c],
                    [coords[0][active], coords[1][active]],
                    order=1,
                    mode="grid-constant",
                    cval=0,
                    prefilter=False,
                )
                for c in range(3)
            ],
            axis=-1,
        )
        rgb = rgb8(rgb + (1 - a) * bg[y : y + h, x : x + w][active])
        original = base[y : y + h, x : x + w][active]
        contrast_counts.append(
            int(
                np.count_nonzero(
                    np.max(np.abs(rgb.astype(np.int16) - original), axis=1) >= 4
                )
            )
        )
        image[y : y + h, x : x + w][active] = rgb
    return (image, min(contrast_counts))


def encode_case(job, work):
    sid, b, mode, t, assets, plan_sha = job
    dest = work / "media" / sid
    if dest.exists():
        q = read(dest / "render.json")
        assert (
            q["plan_sha256"] == plan_sha
            and q["renderer_sha256"] == sha(__file__)
            and (sha(dest / (sid + ".mp4")) == q["video_sha256"])
        )
        return sid
    start, vel, gap, distance = geometry(t, b["N"], b["level"])
    a = assets[b["background"]]
    bg, base = background(a["path"], a["sha256"])
    ids = [b["anchor"]] * b["N"] if mode == "same" else b["distinct"]
    patches = [object_patch(assets[k]["path"], assets[k]["sha256"]) for k in ids]
    dest.parent.mkdir(parents=True, exist_ok=True)
    hashes = []
    stream_hash = hashlib.sha256()
    visible = 56**2
    with TemporaryDirectory(prefix=".render-", dir=dest.parent) as temp:
        output = Path(temp) / "case"
        output.mkdir()
        path = output / (sid + ".mp4")
        with av.open(str(path), "w") as mux:
            codec = mux.add_stream("libx264rgb", rate=FPS)
            codec.width, codec.height, codec.pix_fmt = (W, H, "rgb24")
            codec.codec_context.thread_count = 1
            codec.options = {
                "crf": "0",
                "preset": "ultrafast",
                "threads": "1",
                "x264-params": "bframes=0",
            }
            for k, v in COLOR.items():
                setattr(codec.codec_context, k, v)
            for i in range(F):
                image, count = frame_at(start + vel * (i / FPS), patches, bg, base)
                visible = min(visible, count)
                hashes.append(sha_pixels(image))
                stream_hash.update(image)
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                frame.pts, frame.time_base = (i, Fraction(1, FPS))
                for packet in codec.encode(frame):
                    mux.mux(packet)
            for packet in codec.encode():
                mux.mux(packet)
        q = {
            "sample_id": sid,
            "classic_id": t["classic_id"],
            "plan_sha256": plan_sha,
            "renderer_sha256": sha(__file__),
            "video_sha256": sha(path),
            "stream_sha256": stream_hash.hexdigest(),
            "per_frame": hashes,
            "background_id": b["background"],
            "object_ids_in_point_order": ids,
            "truth": t["truth"],
            "geometry": {"gap_px": gap, "minimum_center_distance_px": distance},
            "minimum_pixels_with_code_difference_ge_4_per_object_frame": visible,
        }
        write_new(output / "render.json", q)
        output.rename(dest)
    return sid
