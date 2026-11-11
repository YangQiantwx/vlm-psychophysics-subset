"""Natural T1 modulation and independent decoded-signal measurements."""

from functools import lru_cache
import numpy as np
from . import natural_common as s


GAIN = 0.5


@lru_cache(4)
def spatial(width=768, height=512):
    yy, xx = np.mgrid[:height, :width]
    x = xx - (width - 1) / 2
    y = yy - (height - 1) / 2
    radius = np.hypot(x, y)
    window = np.where(
        radius <= 192,
        1,
        np.where(radius < 224, 0.5 * (1 + np.cos(np.pi * (radius - 192) / 32)), 0),
    )
    return x, window


def frame_bank(background, sf, m, phase_rad, gain=GAIN):
    """Legacy point-sampled RGB multiplication, with a global headroom gain."""
    height, width = background.shape[:2]
    x, window = spatial(width, height)
    base = gain * background
    phase = 2 * np.pi * sf * x / 448 + phase_rad
    bank = []
    lo = 1.0
    hi = 0.0
    for f in range(12):
        factor = 1 + m * window * np.cos(phase - 2 * np.pi * 5 * f / 60)
        image = base * factor[:, :, None]
        lo = min(lo, float(image.min()))
        hi = max(hi, float(image.max()))
        bank.append(s.rgb8(image))
    assert lo >= -1e-12 and hi <= 1 + 1e-12
    original = s.rgb8(base)
    outside = window == 0
    assert all(np.array_equal(a[outside], original[outside]) for a in bank)
    return bank, {
        "linear_min": lo,
        "linear_max": hi,
        "outside_static": True,
        "clipped_values": 0,
    }


def measure(bank, indices, block, side):
    # Pooled rows improve RGB8 recovery without reading the source image or GT.
    frames = np.array(
        [
            (s.linear(bank[i][240:272, 204:564].astype(float) / 255) @ s.Y).mean(0)
            for i in indices[:12]
        ]
    )
    t = np.arange(12) / 60
    mat = np.column_stack(
        [np.ones(12), np.cos(2 * np.pi * 5 * t), np.sin(2 * np.pi * 5 * t)]
    )
    fit = np.linalg.lstsq(mat, frames, rcond=None)[0]
    z = fit[1] + 1j * fit[2]
    x = np.arange(204, 564) + 0.5 - 384
    phase = np.unwrap(np.angle(z))
    weight = np.abs(z)
    slope, intercept = np.polyfit(x, phase, 1, w=weight)
    direction = 1 if slope > 0 else -1
    sf = abs(slope) * 448 / (2 * np.pi)
    valid = fit[0] > 0.003
    assert (
        valid.sum() >= 100
    ), "Insufficient decoded modulation for independent recovery"
    depth = float(np.median(np.abs(z[valid]) / fit[0, valid]))
    phi = np.deg2rad(7.5 + 15 * block["phase_index"])
    pe = float((direction * intercept - phi + np.pi) % (2 * np.pi) - np.pi)
    q = {
        "measured_sf": float(sf),
        "sf_error": float(sf - block["sf"]),
        "measured_m": depth,
        "m_error": depth - block["m"],
        "phase_error_rad": pe,
        "direction_sign": direction,
        "valid_columns": int(valid.sum()),
        "method": "RGB8 only:32row-pooled temporal LS, spatial complex-phase slope, amplitude/DC",
    }
    q["passed"] = bool(
        direction == (1 if side == "RIGHT" else -1)
        and abs(q["sf_error"]) <= 0.015
        and abs(q["m_error"]) <= 0.003
        and abs(pe) <= 0.05
    )
    assert q["passed"], (block["block_id"], q)
    return q
