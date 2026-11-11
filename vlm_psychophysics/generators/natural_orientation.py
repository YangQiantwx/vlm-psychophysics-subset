"""Natural T3 asset preparation and orientation rendering."""

import hashlib
import math
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .natural_common import (
    Y,
    affine,
    axis,
    angle_diff,
    linear,
    rgb8,
    stamp,
    encode_verify,
    sha,
    read,
    write,
    digest,
)


NAME = "natural-t3-classic-paired-20261102"


def paths(root):
    root = Path(root)
    return (
        root / "plans" / NAME,
        root / "datasets" / NAME,
        root / "runs/stimulus-preparation" / NAME,
    )


def prepare_asset(job):
    root, asset = job
    root = Path(root)
    p, d, r = paths(root)
    out = d / "objects"
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / (asset["id"] + ".json")
    if receipt.exists():
        rec = read(receipt)
        assert rec["source_sha256"] == asset["sha256"]
        assert sha(out / (asset["id"] + "-source.npy")) == rec["source_array_sha256"]
        return rec
    src = root / asset["root_relative_path"]
    assert sha(src) == asset["sha256"]
    rgba = np.asarray(Image.open(src).convert("RGBA"), np.float64) / 255
    alpha = rgba[:, :, 3]
    orientation, center, ratio = axis(alpha)
    scale = math.sqrt(3800 / alpha.sum())
    yy, xx = np.where(alpha > 1e-6)
    radius = float(np.hypot(xx - center[0], yy - center[1]).max() * scale)
    assert ratio >= 4 and radius <= 150
    source = np.dstack([linear(rgba[:, :, :3]) * alpha[:, :, None], alpha])
    target = out / (asset["id"] + "-source.npy")
    np.save(target, source)
    t = math.radians(90 - orientation)
    m = np.eye(3)
    m[:2, :2] = scale * np.array(
        [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]
    )
    m[:2, 2] = 159.5 - m[:2, :2] @ center
    z = affine(source, m)
    a = z[:, :, 3]
    pm = z[:, :, :3]
    assert abs(angle_diff(axis(a)[0], 90)) <= 0.01 and abs(a.sum() / 3800 - 1) <= 0.001
    mu = float((pm @ Y).sum() / a.sum())
    var = float(
        ((np.divide(pm @ Y, a, out=np.zeros_like(a), where=a > 0) - mu) ** 2 * a).sum()
        / a.sum()
    )
    rec = dict(
        asset_id=asset["id"],
        source_sha256=sha(src),
        source_array_sha256=sha(target),
        source_path=asset["root_relative_path"],
        source_axis_deg=orientation,
        center=center.tolist(),
        scale=scale,
        ratio=ratio,
        radius=radius,
        accepted=True,
        features=[
            mu,
            math.sqrt(var),
            math.log(ratio),
            float(
                a.sum()
                / ((xx.max() - xx.min() + 1) * (yy.max() - yy.min() + 1) * scale**2)
            ),
        ],
        area=float(a.sum()),
        alpha_sha256=hashlib.sha256(alpha.tobytes()).hexdigest(),
    )
    write(receipt, rec)
    return rec


def make_patch(job):
    root, ident, theta = job
    p, d, r = paths(root)
    out = d / "patches"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{ident}_{theta:+g}"
    target = out / (tag + ".npy")
    receipt = out / (tag + ".json")
    if receipt.exists():
        rec = read(receipt)
        assert sha(target) == rec["patch_sha256"]
        return rec
    base = read(d / "objects" / (ident + ".json"))
    source = np.load(d / "objects" / (ident + "-source.npy"), mmap_mode="r")
    t = math.radians(90 - base["source_axis_deg"] + theta)
    m = np.eye(3)
    m[:2, :2] = base["scale"] * np.array(
        [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]
    )
    m[:2, 2] = 159.5 - m[:2, :2] @ np.array(base["center"])
    attempts = []
    for samples in (8, 16, 32):
        z = affine(source, m, samples=samples)
        a = z[:, :, 3]
        observed = axis(a)[0]
        error = angle_diff(observed, 90 + theta)
        attempts.append(
            dict(
                samples=samples,
                axis_error_deg=error,
                area_error_relative=float(a.sum() / 3800 - 1),
            )
        )
        if abs(error) <= 0.01 and abs(a.sum() / 3800 - 1) <= 0.001:
            break
    assert abs(error) <= 0.01 and abs(a.sum() / 3800 - 1) <= 0.001, (
        ident,
        theta,
        attempts,
    )
    assert not any(np.any(x) for x in (a[0], a[-1], a[:, 0], a[:, -1]))
    np.save(target, z)
    rec = dict(
        asset_id=ident,
        theta=theta,
        source_sha256=base["source_sha256"],
        matrix=m.tolist(),
        measured_axis_deg=observed,
        axis_error_deg=error,
        alpha_area=float(a.sum()),
        integration_attempts=attempts,
        patch_sha256=sha(target),
        implementation_sha256=sha(__file__),
    )
    write(receipt, rec)
    return rec


def render_one(job):
    root, row = job
    root = Path(root)
    p, d, r = paths(root)
    q = d / "qa" / (row["sample_id"] + ".json")
    target = d / "videos" / row["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if q.exists():
        old = read(q)
        assert (
            old["allocation_row_sha256"] == digest(row)
            and sha(target) == old["video_sha256"]
        )
        return dict(sample_id=row["sample_id"], existing=True)
    start = time.monotonic()
    b = np.load(root / row["background_prepared_path"], mmap_mode="r")
    theta = row["render_theta_deg"]
    ident = row["object_id"]
    tags = [f"{ident}_{v:+g}" for v in (0.0, theta)]
    z0, zt = [np.load(d / "patches" / (tag + ".npy"), mmap_mode="r") for tag in tags]
    metas = [read(d / "patches" / (tag + ".json")) for tag in tags]
    assert all(m["source_sha256"] == row["object_source_sha256"] for m in metas)
    event = b.copy()
    left, right = (z0, zt) if row["truth"] == "LEFT" else (zt, z0)
    stamp(event, left, 224, 304)
    stamp(event, right, 544, 304)
    bank = [rgb8(b), rgb8(event)]
    roi = np.zeros((512, 768), bool)
    roi[144:464, 64:384] = left[:, :, 3] > 0
    roi[144:464, 384:704] = right[:, :, 3] > 0
    assert np.array_equal(bank[0][~roi], bank[1][~roi])
    onset = row["onset_frame"]
    indices = [int(onset <= f < onset + 30) for f in range(180)]
    receipt = encode_verify(target, bank, indices)
    # Independent direction check uses the covariance cross term, not the renderer sign.
    a = zt[:, :, 3]
    yy, xx = np.indices(a.shape, dtype=float)
    cx = (a * xx).sum() / a.sum()
    cy = (a * yy).sum() / a.sum()
    cross = float((a * (xx - cx) * (yy - cy)).sum())
    physical = 0 if row["angle_deg"] == 90 else (1 if cross > 0 else -1)
    assert physical == row["physical_tilt_sign"], (
        row["sample_id"],
        physical,
        row["physical_tilt_sign"],
    )
    observed = abs(
        angle_diff(metas[1]["measured_axis_deg"], metas[0]["measured_axis_deg"])
    )
    assert abs(observed - row["angle_deg"]) <= 0.02
    rec = {
        **receipt,
        "status": "PASS",
        "sample_id": row["sample_id"],
        "classic_sample_id": row["classic_sample_id"],
        "allocation_row_sha256": digest(row),
        "patches": metas,
        "physical_tilt_sign_measured": physical,
        "actual_angle_deg": observed,
        "onset_frame": onset,
        "offset_frame": onset + 30,
        "unchanged_background_outside_objects": True,
        "elapsed_seconds": time.monotonic() - start,
        "implementation_sha256": sha(__file__),
    }
    write(q, rec)
    if row.get("representative"):
        out = d / "review"
        out.mkdir(exist_ok=True)
        for i in [onset - 1, onset, onset + 29, onset + 30]:
            Image.fromarray(bank[indices[i]]).save(
                out / f'{row["sample_id"]}_f{i:03}.png'
            )
    return dict(sample_id=row["sample_id"], seconds=rec["elapsed_seconds"])
