"""Linear-light transforms, source geometry and lossless RGB8 encoding."""

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates


Y = np.array([0.2126, 0.7152, 0.0722])


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    q.replace(p)


def digest(d):
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def rgb8(x):
    assert np.isfinite(x).all() and x.min() >= -1e-10 and x.max() <= 1 + 1e-10
    x = np.clip(x, 0, 1)
    return np.rint(
        255 * np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
    ).astype(np.uint8)


def axis(a):
    yy, xx = np.indices(a.shape, dtype=float)
    mass = a.sum()
    cx = (a * xx).sum() / mass
    cy = (a * yy).sum() / mass
    xx -= cx
    yy -= cy
    cov = (
        np.array(
            [
                [(a * xx * xx).sum(), (a * xx * yy).sum()],
                [(a * xx * yy).sum(), (a * yy * yy).sum()],
            ]
        )
        / mass
    )
    val, vec = np.linalg.eigh(cov)
    v = vec[:, -1]
    return (
        float(np.degrees(np.arctan2(v[1], v[0]))),
        np.array([cx, cy]),
        float(val[-1] / val[0]),
    )


def angle_diff(a, b):
    return (a - b + 90) % 180 - 90


def affine(source, matrix, size=320, samples=8):
    """Inverse-map bilinear reconstruction integrated over an 8x8 pixel grid."""
    inv = np.linalg.inv(matrix)
    yy, xx = np.indices((size, size), dtype=float)
    out = np.zeros((size, size, 4), dtype=float)
    for dy in (np.arange(samples) + 0.5) / samples - 0.5:
        for dx in (np.arange(samples) + 0.5) / samples - 0.5:
            sx = inv[0, 0] * (xx + dx) + inv[0, 1] * (yy + dy) + inv[0, 2]
            sy = inv[1, 0] * (xx + dx) + inv[1, 1] * (yy + dy) + inv[1, 2]
            for c in range(4):
                out[:, :, c] += map_coordinates(
                    source[:, :, c],
                    [sy, sx],
                    order=1,
                    mode="constant",
                    cval=0,
                    prefilter=False,
                )
    return out / samples**2


def stamp(bg, z, cx, cy):
    h, w = z.shape[:2]
    x = cx - w // 2
    y = cy - h // 2
    bg[y : y + h, x : x + w] = (
        z[:, :, :3] + (1 - z[:, :, 3, None]) * bg[y : y + h, x : x + w]
    )


def encode_verify(path, bank, indices):
    import av, cv2

    cv2.setNumThreads(1)
    path = Path(path)
    temp = path.with_suffix(".partial.mp4")
    out = av.open(str(temp), "w")
    stream = out.add_stream("libx264rgb", rate=60)
    stream.width = 768
    stream.height = 512
    stream.pix_fmt = "rgb24"
    stream.options = {
        "crf": "0",
        "preset": "ultrafast",
        "threads": "1",
        "x264-params": "ref=16:scenecut=0",
    }
    stream.codec_context.thread_count = 1
    expected = hashlib.sha256()
    for n, i in enumerate(indices):
        raw = bank[i]
        expected.update(raw.tobytes())
        frame = av.VideoFrame.from_ndarray(raw, format="rgb24")
        frame.pts = n
        frame.time_base = Fraction(1, 60)
        for packet in stream.encode(frame):
            out.mux(packet)
    for packet in stream.encode():
        out.mux(packet)
    out.close()
    cap = cv2.VideoCapture(str(temp), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    actual = hashlib.sha256()
    n = 0
    assert abs(cap.get(cv2.CAP_PROP_FPS) - 60) < 1e-6
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        assert n < len(indices) and np.array_equal(rgb, bank[indices[n]]), (
            path,
            n,
            "decoded RGB differs",
        )
        assert abs(cap.get(cv2.CAP_PROP_POS_MSEC) - n * 1000 / 60) < 0.001, (
            path,
            n,
            "frame timestamp differs",
        )
        actual.update(rgb.tobytes())
        n += 1
    cap.release()
    assert n == 180 and actual.hexdigest() == expected.hexdigest()
    temp.replace(path)
    return {
        "video_sha256": sha(path),
        "RGB_stream_sha256": actual.hexdigest(),
        "frames_byte_equal": n,
        "timestamps_checked": n,
        "bytes": path.stat().st_size,
    }
