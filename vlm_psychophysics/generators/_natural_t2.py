"""Frozen Natural T2 source transforms and RGB8 codec routines."""

from fractions import Fraction
import math

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.sparse import csr_matrix

Y = np.array([0.2126, 0.7152, 0.0722])
CENTERS = ((224, 302), (544, 302))


def linear(a):
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def area_weights(source, dest):
    """Exact overlap of output boxes with piecewise-constant source pixels."""
    edges = np.linspace(0, source, dest + 1)
    ii, jj, ww = ([], [], [])
    for i in range(dest):
        for j in range(
            int(np.floor(edges[i])), min(source, int(np.ceil(edges[i + 1])))
        ):
            v = max(0.0, min(edges[i + 1], j + 1) - max(edges[i], j))
            if v:
                ii.append(i)
                jj.append(j)
                ww.append(v / (edges[i + 1] - edges[i]))
    weights = csr_matrix((ww, (ii, jj)), shape=(dest, source))
    assert np.max(np.abs(np.asarray(weights.sum(axis=1)).ravel() - 1)) < 1e-12
    return weights


def background(path, width=768, height=512):
    im = np.asarray(Image.open(path).convert("RGB"), np.float64) / 255
    h, w = im.shape[:2]
    if w / h > width / height:
        cw = int(h * width / height)
        x = (w - cw) // 2
        crop = (x, 0, x + cw, h)
    else:
        ch = int(w * height / width)
        y = (h - ch) // 2
        crop = (0, y, w, y + ch)
    x0, y0, x1, y1 = crop
    im = linear(im[y0:y1, x0:x1])
    wy, wx = (area_weights(im.shape[0], height), area_weights(im.shape[1], width))
    out = np.stack([(wx @ (wy @ im[:, :, k]).T).T for k in range(3)], axis=-1)
    return (
        out,
        {
            "crop_xyxy": list(crop),
            "resize": "FP64 linear-RGB exact box overlap",
            "input_dimensions": [w, h],
            "output_dimensions": [width, height],
        },
    )


def axis(alpha):
    y, x = np.indices(alpha.shape, dtype=np.float64)
    mass = alpha.sum()
    cx = float((alpha * x).sum() / mass)
    cy = float((alpha * y).sum() / mass)
    x -= cx
    y -= cy
    cov = (
        np.array(
            [
                [(alpha * x * x).sum(), (alpha * x * y).sum()],
                [(alpha * x * y).sum(), (alpha * y * y).sum()],
            ]
        )
        / mass
    )
    values, vectors = np.linalg.eigh(cov)
    v = vectors[:, -1]
    return (
        math.degrees(math.atan2(v[1], v[0])),
        (cx, cy),
        float(values[-1] / max(values[0], 1e-12)),
    )


def sample_affine(source, matrix, size=256, samples=8):
    """One source-to-output affine, integrated on a uniform subpixel grid."""
    inv = np.linalg.inv(matrix)
    y, x = np.indices((size, size), dtype=np.float64)
    out = np.zeros((size, size, source.shape[2]), np.float64)
    for dy in (np.arange(samples) + 0.5) / samples - 0.5:
        for dx in (np.arange(samples) + 0.5) / samples - 0.5:
            sx = inv[0, 0] * (x + dx) + inv[0, 1] * (y + dy) + inv[0, 2]
            sy = inv[1, 0] * (x + dx) + inv[1, 1] * (y + dy) + inv[1, 2]
            for c in range(source.shape[2]):
                out[:, :, c] += map_coordinates(
                    source[:, :, c],
                    [sy, sx],
                    order=1,
                    mode="constant",
                    cval=0,
                    prefilter=False,
                )
    return out / samples**2


def rgb8(v):
    assert np.isfinite(v).all() and v.min() >= -1e-12 and (v.max() <= 1 + 1e-12)
    v = np.clip(v, 0, 1)
    return np.rint(
        255 * np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)
    ).astype(np.uint8)


def stamp(bg, pm, a, side):
    cx, cy = CENTERS[0 if side == "LEFT" else 1]
    half = a.shape[0] // 2
    s = (slice(cy - half, cy + half), slice(cx - half, cx + half))
    bg[s] = pm + (1 - a[..., None]) * bg[s]


def encode(path, baseline, event, onset, offset):
    import av

    out = av.open(str(path), "w")
    stream = out.add_stream("libx264rgb", rate=60)
    stream.width = 768
    stream.height = 512
    stream.pix_fmt = "rgb24"
    stream.codec_context.thread_count = 1
    stream.options = {
        "crf": "0",
        "preset": "veryfast",
        "x264-params": "keyint=180:min-keyint=180:scenecut=0:bframes=0",
    }
    for i in range(180):
        frame = av.VideoFrame.from_ndarray(
            event if onset <= i < offset else baseline, format="rgb24"
        )
        frame.pts = i
        frame.time_base = Fraction(1, 60)
        for packet in stream.encode(frame):
            out.mux(packet)
    for packet in stream.encode():
        out.mux(packet)
    out.close()
