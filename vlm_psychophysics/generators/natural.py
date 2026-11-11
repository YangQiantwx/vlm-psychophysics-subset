"""Render Natural T1–T4 trials from their frozen source records."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from . import natural_motion as t1
from . import natural_orientation as t3

from . import _natural_t2 as t2
from . import _natural_t4 as t4


T4_TABLE = "sources/natural/t4/combinations.json"
T2_TABLE = "sources/natural/t2/combinations.json"
T3_ROOT = "sources/natural/t3/"
BACKGROUND_ROOT = "natural/t1/datasets/natural-t13-20261029/backgrounds/"


def background_path(data, asset_id, expected=None):
    if expected is None:
        expected = data.json(T4_TABLE)["assets"][asset_id]["sha256"]
    return data.file(f"{BACKGROUND_ROOT}{asset_id}.npy", expected)


def render_t1(data, row, destination):
    p = row["parameters"]
    path = background_path(data, row["background_id"])
    bank, _ = t1.frame_bank(
        np.load(path, allow_pickle=False),
        p["sf"],
        p["m"],
        np.deg2rad(7.5 + 15 * p["phase_index"]),
    )
    sign = 1 if row["truth"] == "RIGHT" else -1
    frames = [sign * frame % 12 for frame in range(180)]
    t1.measure(bank, frames, p, row["truth"])
    t1.s.encode_verify(destination, bank, frames)


@lru_cache(maxsize=4)
def t2_background(path):
    return t2.background(path)[0]


def t2_matrix(asset, alpha):
    """Placement of the replacement spoon, which has no stored affine matrix."""
    axis, _, ratio = t2.axis(alpha)
    yy, xx = np.where(alpha > 1e-6)
    points = asset.get("axis_points_px")
    if points:
        dx, dy = np.subtract(points[1], points[0])
        theta = 90 - float(np.degrees(np.arctan2(dy, dx)))
    elif ratio >= 4:
        theta = 90 - axis
    else:
        theta = 0.0
    radians = np.deg2rad(theta)
    rotation = np.array(
        [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
    )
    rotated = rotation @ np.stack([xx, yy])
    low, high = rotated.min(axis=1), rotated.max(axis=1)
    area_scale = np.sqrt(5118.7529296875 / alpha.sum())
    scale = min(area_scale, 300 / (high[0] - low[0] + 2), 380 / (high[1] - low[1] + 2))
    matrix = np.eye(3)
    matrix[:2, :2] = scale * rotation
    matrix[:2, 2] = 191.5 - scale * (low + high) / 2
    return matrix


@lru_cache(maxsize=8)
def t2_object(path, matrix, axis_points):
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), np.float64) / 255
    alpha = rgba[..., 3:]
    if matrix is None:
        matrix = t2_matrix({"axis_points_px": axis_points}, alpha[..., 0])
    source = np.concatenate((t2.linear(rgba[..., :3]) * alpha, alpha), axis=-1)
    return t2.sample_affine(source, np.asarray(matrix), size=384, samples=8)


def t2_asset(data, asset):
    path = asset.get("repository_path")
    if path is None:
        path = f"sources/natural/t2/icc/{Path(asset['bundle_path']).name}"
    return data.file(path, asset["sha256"])


def index_t2(table):
    return {
        case["sample_id"]: (row, case)
        for row in table["combinations"]
        for case in row["cases"]
    }


def render_t2(data, row, destination):
    table = data.json(T2_TABLE)
    trial, case = data.index(T2_TABLE, index_t2)[row["sample_id"]]
    if (
        case["truth"] != row["truth"]
        or case["decoded_stream_sha256"] != row["decoded_stream_sha256"]
    ):
        raise ValueError(f"Natural T2 metadata differs for {row['sample_id']}")
    assets = table["assets"]
    bg_asset = assets[trial["background_id"]]
    obj_asset = assets[trial["object_id"]]
    bg = t2_background(t2_asset(data, bg_asset)).copy()
    matrix = obj_asset.get("source_to_output_matrix")
    axes = obj_asset.get("axis_points_px")
    obj = t2_object(
        t2_asset(data, obj_asset),
        tuple(map(tuple, matrix)) if matrix is not None else None,
        tuple(map(tuple, axes)) if axes is not None else None,
    )
    pm, alpha = obj[..., :3], obj[..., 3]
    delta = trial["delta_Y"]
    gain = delta / (1 - (pm @ t2.Y).sum() / alpha.sum())
    if not 0 < gain < 1:
        raise ValueError(f"Invalid Natural T2 brightening gain for {row['sample_id']}")
    for side in ("LEFT", "RIGHT"):
        t2.stamp(bg, pm, alpha, side)
    base = t2.rgb8(bg)
    cx, cy = t2.CENTERS[("LEFT", "RIGHT").index(case["truth"])]
    bg[cy - 192 : cy + 192, cx - 192 : cx + 192] += gain * (alpha[..., None] - pm)
    event = t2.rgb8(bg)
    t2.encode(destination, base, event, trial["onset_frame"], trial["offset_frame"])


def index_t3(allocation):
    return {row["sample_id"]: row for row in allocation["rows"]}


def index_t3_objects(assets):
    return {asset["id"]: asset for asset in assets["objects"]}


def index_t3_sources(assets):
    return {asset["asset_id"]: asset for asset in assets["files"]}


def link_verified(target, source):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve())


def render_t3(data, row, destination, work):
    trial = data.index(T3_ROOT + "allocation.json", index_t3)[row["sample_id"]]
    object_id = trial["object_id"]
    asset = data.index(T3_ROOT + "assets.json", index_t3_objects)[object_id]
    source = data.index(T3_ROOT + "asset-files.json", index_t3_sources)[object_id]
    bg = trial["background_prepared_path"]
    link_verified(
        work / bg, data.file(f"natural/t1/{bg}", trial["background_prepared_sha256"])
    )
    link_verified(
        work / source["root_path"],
        data.file(f"natural/t3/{source['repo_path']}", source["sha256"]),
    )
    _, output, _ = t3.paths(work)
    (output / "qa").mkdir(parents=True, exist_ok=True)
    t3.prepare_asset((work, asset))
    for angle in (0.0, trial["render_theta_deg"]):
        t3.make_patch((work, object_id, angle))
    t3.render_one((work, trial))
    (output / "videos" / trial["path"]).replace(destination)


def index_t4(plan):
    return {case[0]: case for case in t4.cases(plan)}


def render_t4(data, row, destination, work):
    plan = data.json(T4_TABLE)
    sample_id = row["sample_id"]
    case = data.index(T4_TABLE, index_t4)[sample_id]
    _, block, mode, _ = case
    ids = [block["anchor"]] if mode == "same" else block["distinct"]
    bg = block["background"]
    assets = {}
    for asset_id in (*ids, bg):
        spec = plan["assets"][asset_id]
        if asset_id == bg:
            path = background_path(data, asset_id, spec["sha256"])
        else:
            path = data.file(
                f"sources/natural/t4/objects/{asset_id}-T4.npy", spec["sha256"]
            )
        assets[asset_id] = {**spec, "path": path}
    t4.encode_case((*case, assets, data.sha(T4_TABLE)), work)
    generated = work / "media" / sample_id
    receipt = t4.read(generated / "render.json")
    for image in row["static"]["images"]:
        if receipt["per_frame"][image["frame_index"]] != image["pixel_sha256"]:
            raise ValueError(f"Natural T4 endpoint differs for {sample_id}")
    (generated / f"{sample_id}.mp4").replace(destination)
