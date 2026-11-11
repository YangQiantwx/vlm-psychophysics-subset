"""Image-only scientific checks, kept separate from nominal labels."""

from __future__ import annotations

import numpy as np
from . import raster


def linear_y(codes):
    rgb = np.asarray(codes, dtype=np.float64) / 65535.0
    return raster.linear(rgb) @ np.array([0.2126, 0.7152, 0.0722])


def source_state(row, baseline, event, coverage):
    p = row["source_parameters"]
    if baseline.dtype != np.uint16 or event.dtype != np.uint16:
        raise ValueError("Scientific source must be RGB uint16")
    if baseline.shape != (p["height"], p["width"], 3) or event.shape != baseline.shape:
        raise ValueError("Source shape mismatch")
    if row["task"] == "T2":
        a = coverage
        dy = linear_y(event) - linear_y(baseline)
        opaque, roi = a >= 1 - 1e-12, a > 0
        if not np.any(opaque):
            raise ValueError("No opaque core for photometric validation")
        core = float(dy[opaque].mean())
        slope = float(np.dot(a[roi], dy[roi]) / np.dot(a[roi], a[roi]))
        result = {
            "task": "T2",
            "core_increment": core,
            "core_max_error": float(np.max(abs(dy[opaque] - p["delta_Y"]))),
            "alpha_regression_increment": slope,
            "alpha_regression_error": abs(slope - p["delta_Y"]),
            "ROI_max_residual": float(np.max(abs(dy[roi] - a[roi] * p["delta_Y"]))),
            "changed_outside_target": int(
                np.count_nonzero(event[~roi] != baseline[~roi])
            ),
        }
        if max(result["core_max_error"], result["alpha_regression_error"]) >= 0.0001:
            raise ValueError(("T2 photometric tolerance", result))
        if result["changed_outside_target"]:
            raise ValueError(("T2 changed background/reference", result))
        return result
    if (p["background_Y"], p["foreground_Y"]) != (raster.BG, raster.FG):
        raise ValueError("Pixel-axis estimators require the declared BG/FG calibration")
    axes = []
    for index, side in enumerate(("LEFT", "RIGHT")):
        cx, cy = p["centers_xy"][index]
        q = event[cy - 128 : cy + 128, cx - 128 : cx + 128, 0]
        nominal = (
            0.0 if side == p["reference_side"] else p["angle_deg"] * p["tilt_sign"]
        )
        # Neither estimator receives nominal or renderer geometry.
        pca = raster.pca_axis(raster.decode(q, 16))
        edge = raster.quantized_edge_axis(q, 16)
        if edge.get("status") != "ok" or not edge.get("optimizer_success"):
            raise ValueError(("Image-only edge fit failed", side, edge))
        result = {
            "side": side,
            "nominal_deg": nominal,
            "PCA_deg": pca,
            "center_xy": [cx, cy],
            "bar_wh": p["bar_wh"],
            "background_Y": p["background_Y"],
            "foreground_Y": p["foreground_Y"],
            "PCA_error_deg": raster.angle_error(pca, nominal),
            "halfplane_deg": edge["axis_deg"],
            "halfplane_error_deg": raster.angle_error(edge["axis_deg"], nominal),
            "conditional_code_interval_feasible": edge[
                "quantization_interval_feasible"
            ],
        }
        if max(result["PCA_error_deg"], result["halfplane_error_deg"]) >= 0.001:
            raise ValueError(("T3 pixel-axis tolerance", result))
        axes.append(result)
    return {"task": "T3", "axes": axes}


def source_panel(records):
    unique = {}
    for r in records:
        if r["task"] != "T3":
            continue
        for a in r["axes"]:
            key = (
                tuple(a["center_xy"]),
                a["nominal_deg"],
                tuple(a["bar_wh"]),
                a["background_Y"],
                a["foreground_Y"],
            )
            if key in unique and unique[key] != a:
                raise ValueError(
                    "Repeated geometry produced different image-axis measurements"
                )
            unique[key] = a
    axes = list(unique.values())
    if axes:
        for field in ("PCA_error_deg", "halfplane_error_deg"):
            if float(np.mean([a[field] for a in axes])) >= 0.0002:
                raise ValueError(("T3 panel mean error", field))
    return {
        "status": "PASS",
        "unique_states": len(records),
        "T3_unique_geometries": len(axes),
        "T3_PCA_MAE_deg": (
            float(np.mean([a["PCA_error_deg"] for a in axes])) if axes else None
        ),
        "T3_halfplane_MAE_deg": (
            float(np.mean([a["halfplane_error_deg"] for a in axes])) if axes else None
        ),
    }
