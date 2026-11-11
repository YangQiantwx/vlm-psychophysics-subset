"""Frozen Classic T1 and strict Classic T4 raster and codec routines."""

from fractions import Fraction
import math
import os
from functools import lru_cache

import av
import numpy as np

WIDTH, HEIGHT, FPS, FRAMES = 768, 512, 60, 180
MEAN = 0.31
DIAMETER = 56.0
BG, FG = 0.10, 0.82
OPTIONS = {"coder": "1", "context": "1", "level": "3", "slicecrc": "1", "slices": "4"}
COLOR = {"color_range": 2, "colorspace": 0, "color_primaries": 2, "color_trc": 2}


@lru_cache(maxsize=8)
def basis(path, sf):
    with np.load(path, allow_pickle=False) as data:
        if float(data["sf"]) != sf:
            raise ValueError("Basis spatial frequency differs")
        return data["cos"], data["sin"]


def token(x):
    return format(float(x), ".10g").replace(".", "p")


def quantize(y):
    if y.dtype != np.float64 or np.min(y) < -1e-13 or np.max(y) > 1 + 1e-13:
        raise ValueError("Source is not valid FP64 linear luminance")
    encoded = np.where(
        y <= 0.0031308, 12.92 * y, 1.055 * np.power(np.maximum(y, 0.0), 1 / 2.4) - 0.055
    )
    return np.rint(encoded * 65535.0).astype(np.uint16)


def states(row, basis_file):
    p = row["params"]
    c, s = basis(basis_file, p["sf_cycles_per_aperture"])
    result = []
    # Index arithmetic makes the prescribed 24-phase bank exact and avoids
    # roundoff differences between repeated cycles. Each video has 12 states.
    for f in range(12):
        index = (row["phase_index"] - 2 * p["direction_sign"] * f) % 24
        phase = math.radians(7.5 + 15 * index)
        y = MEAN * (
            1 + p["modulation_depth"] * (c * math.cos(phase) - s * math.sin(phase))
        )
        result.append(quantize(y))
    return result


def encode(path, bank):
    with av.open(str(path), "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=FPS)
        stream.width, stream.height = WIDTH, HEIGHT
        stream.pix_fmt = "gbrp16le"
        stream.codec_context.thread_count = 1
        stream.codec_context.colorspace = 0
        stream.codec_context.color_range = 2
        stream.options = {"level": "3", "coder": "1", "context": "1", "slicecrc": "1"}
        rgb_bank = [np.repeat(v[..., None], 3, axis=2) for v in bank]
        for i in range(FRAMES):
            frame = av.VideoFrame.from_ndarray(rgb_bank[i % 12], format="rgb48le")
            frame.pts, frame.time_base = i, Fraction(1, FPS)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    with path.open("rb") as f:
        os.fsync(f.fileno())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def circle_box_area(x0, x1, y0, y1, radius=DIAMETER / 2):
    """Vectorized exact circle-box integral, split at all circle/box crossings."""
    x0, x1, y0, y1 = np.broadcast_arrays(x0, x1, y0, y1)
    lo, hi = np.maximum(x0, -radius), np.minimum(x1, radius)
    q0 = np.sqrt(np.maximum(0, radius * radius - y0 * y0))
    q1 = np.sqrt(np.maximum(0, radius * radius - y1 * y1))
    cuts = np.sort(
        np.clip(np.stack((lo, hi, -q0, q0, -q1, q1), -1), lo[..., None], hi[..., None]),
        -1,
    )
    a, b = cuts[..., :-1], cuts[..., 1:]
    h = np.sqrt(np.maximum(0, radius * radius - ((a + b) / 2) ** 2))
    top_flat, bottom_flat = y1[..., None] <= h, y0[..., None] >= -h
    flat = np.where(top_flat, y1[..., None], 0) - np.where(
        bottom_flat, y0[..., None], 0
    )
    curved = (~top_flat).astype(int) + (~bottom_flat).astype(int)

    def primitive(x):
        return 0.5 * (
            x * np.sqrt(np.maximum(0, radius * radius - x * x))
            + radius * radius * np.arcsin(np.clip(x / radius, -1, 1))
        )

    integrals = flat * (b - a) + curved * (primitive(b) - primitive(a))
    valid = np.minimum(y1[..., None], h) > np.maximum(y0[..., None], -h)
    return np.clip(np.sum(np.where(valid, integrals, 0), -1), 0, 1)


def disc_patch(cx, cy):
    radius = DIAMETER / 2
    x0, y0 = math.floor(cx - radius), math.floor(cy - radius)
    x1, y1 = math.ceil(cx + radius), math.ceil(cy + radius)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    far = (
        np.maximum(abs(xx - cx), abs(xx + 1 - cx)) ** 2
        + np.maximum(abs(yy - cy), abs(yy + 1 - cy)) ** 2
    )
    near = (
        np.maximum(np.maximum(xx - cx, cx - xx - 1), 0) ** 2
        + np.maximum(np.maximum(yy - cy, cy - yy - 1), 0) ** 2
    )
    alpha = (far <= radius**2).astype(np.float64)
    boundary = (near < radius**2) & (far > radius**2)
    bx, by = xx[boundary] - cx, yy[boundary] - cy
    alpha[boundary] = circle_box_area(bx, bx + 1, by, by + 1)
    require(
        abs(alpha.sum() - math.pi * radius**2) < 1e-7,
        "Disk pixel integral does not conserve area",
    )
    return x0, y0, alpha


class MixedShapes:
    """Pixel-area rasterization of the published, fixed-orientation shape polygons."""

    def __init__(self, definitions):
        import shapely
        from shapely.geometry import Polygon

        self.shapely = shapely
        self.definitions = definitions
        self.geometry = {}
        for name, shape in definitions.items():
            if name == "circle":
                continue
            polygon = Polygon(shape["outer"], shape["holes"])
            inside, outside = polygon.buffer(-0.8), polygon.buffer(0.8)
            shapely.prepare(inside)
            shapely.prepare(outside)
            self.geometry[name] = (polygon, inside, outside, polygon.bounds)

    def patch(self, name, cx, cy):
        if name == "circle":
            return disc_patch(cx, cy)
        shapely = self.shapely
        polygon, inside, outside, bounds = self.geometry[name]
        x0, y0 = math.floor(cx + bounds[0]), math.floor(cy + bounds[1])
        x1, y1 = math.ceil(cx + bounds[2]), math.ceil(cy + bounds[3])
        yy, xx = np.mgrid[y0:y1, x0:x1]
        x, y = xx - cx, yy - cy
        alpha = shapely.contains_xy(inside, x + 0.5, y + 0.5).astype(float)
        edge = shapely.contains_xy(outside, x + 0.5, y + 0.5) & (alpha == 0)
        cells = shapely.box(x[edge], y[edge], x[edge] + 1, y[edge] + 1)
        alpha[edge] = shapely.area(shapely.intersection(cells, polygon, grid_size=0))
        require(
            alpha.min() >= -1e-12 and alpha.max() <= 1 + 1e-10, "Invalid shape coverage"
        )
        require(
            abs(alpha.sum() - self.definitions[name]["area_px2"]) < 2e-7,
            "Shape pixel integral does not conserve area",
        )
        return x0, y0, alpha


def encode_strict(video, track, *, shape_ids=None, definitions=None):
    raster = MixedShapes(definitions) if shape_ids is not None else None
    if raster is not None:
        require(len(shape_ids) == track.shape[1], "Shape and trajectory counts differ")
    area = (
        sum(definitions[name]["area_px2"] for name in shape_ids)
        if raster is not None
        else track.shape[1] * np.pi * 28**2
    )
    background = int(np.rint((1.055 * BG ** (1 / 2.4) - 0.055) * 65535))
    with av.open(str(video), "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=60)
        stream.width, stream.height, stream.pix_fmt = 768, 512, "gbrp16le"
        stream.codec_context.thread_count = 1
        stream.time_base = Fraction(1, 60)
        stream.options = OPTIONS
        for key, value in COLOR.items():
            setattr(stream.codec_context, key, value)
        for index, points in enumerate(track):
            coverage = np.zeros((512, 768), np.float64)
            for point_index, (cx, cy) in enumerate(points):
                x, y, alpha = (
                    raster.patch(shape_ids[point_index], cx, cy)
                    if raster is not None
                    else disc_patch(cx, cy)
                )
                h, w = alpha.shape
                assert 0 <= x <= 768 - w and 0 <= y <= 512 - h, (video, index, x, y)
                # Disjoint disks can both cover part of the same edge pixel.
                coverage[y : y + h, x : x + w] += alpha
            assert coverage.max() <= 1 + 1e-10, (video, index, coverage.max())
            assert abs(coverage.sum() - area) < (
                3e-6 if raster is not None else 2e-6
            ), (
                video,
                index,
            )
            linear = BG + (FG - BG) * np.minimum(coverage, 1)
            gray = np.rint((1.055 * linear ** (1 / 2.4) - 0.055) * 65535).astype("<u2")
            assert all(
                np.all(edge == background)
                for edge in (gray[0], gray[-1], gray[:, 0], gray[:, -1])
            )
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb48le")
            frame.pts, frame.time_base = index, Fraction(1, 60)
            for key, value in COLOR.items():
                if hasattr(frame, key):
                    setattr(frame, key, value)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
