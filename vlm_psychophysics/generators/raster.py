"""FP64 box-filter rasterization and independent pixel-axis estimators.

Continuous pixel squares are [x,x+1] x [y,y+1]. The rectangle is rigidly
rotated; positive angles lean its lower end right. Area integration implements
standard analytic antialiasing, independently checked against GEOS, Manson's
FilterBox source and Linux Psychtoolbox captures. No toolkit-equivalence claim
is made outside the tested straight-edge, box-filter cases.
"""

from __future__ import annotations
import math
import numpy as np
import shapely
from shapely.geometry import Polygon
from scipy.integrate import quad
from scipy.optimize import linprog, least_squares, minimize_scalar, brentq

BG, FG = 0.10, 0.82


def angle_error(a, b):
    return abs((a - b + 90) % 180 - 90)


def vertices(theta, cx=128.0, cy=128.0, width=20.0, height=190.0):
    t = math.radians(theta)
    c, s = math.cos(t), math.sin(t)
    return np.array(
        [
            [cx + c * u + s * v, cy - s * u + c * v]
            for u, v in [
                (-width / 2, -height / 2),
                (width / 2, -height / 2),
                (width / 2, height / 2),
                (-width / 2, height / 2),
            ]
        ]
    )


def _hinge_integral(m, b, ya, yb):
    """Integral of max(m*y+b,0), vectorized in b without cancellation."""
    p = m * ya + b
    q = m * yb + b
    d = yb - ya
    if abs(m) < 1e-15:
        return np.maximum(p, 0) * d
    both = (p >= 0) & (q >= 0)
    cross = p * q < 0
    out = np.where(both, (p + q) * d / 2, 0.0)
    out = np.where(cross, np.maximum(p, q) ** 2 / (2 * abs(m)), out)
    return out


def strip_coverage(
    theta,
    dx=0.0,
    dy=0.0,
    shape=(256, 256),
    center=(128.0, 128.0),
    width=20.0,
    height=190.0,
    pixel_indices=None,
):
    """Exact box coverage by piecewise-affine horizontal boundary integration.

    Each interval between rectangle-vertex y coordinates has exactly two active
    edges. The overlap length with [x,x+1] equals a signed sum of four hinges.
    This is an independent integration algorithm, not polygon/pixel clipping.
    """
    p = vertices(theta, center[0] + dx, center[1] + dy, width, height)
    if pixel_indices is None:
        lo = np.maximum(0, np.floor(p.min(0)).astype(int))
        hi = np.minimum(shape[::-1], np.ceil(p.max(0)).astype(int))
        yy, xx = np.mgrid[lo[1] : hi[1], lo[0] : hi[0]]
        yy = yy.ravel()
        xx = xx.ravel()
    else:
        yy, xx = np.asarray(pixel_indices).T
    result = np.zeros(len(xx), np.float64)
    for ya, yb in zip(sorted(set(p[:, 1]))[:-1], sorted(set(p[:, 1]))[1:]):
        if yb - ya < 1e-13:
            continue
        mid = (ya + yb) / 2
        edges = []
        for a, b in zip(p, np.roll(p, -1, axis=0)):
            if min(a[1], b[1]) < mid < max(a[1], b[1]):
                m = (b[0] - a[0]) / (b[1] - a[1])
                k = a[0] - m * a[1]
                edges.append((m, k))
        edges.sort(key=lambda v: v[0] * mid + v[1])
        assert len(edges) == 2, (theta, ya, yb, edges)
        ids = np.flatnonzero((yy < yb) & (yy + 1 > ya))
        l = np.maximum(yy[ids], ya)
        h = np.minimum(yy[ids] + 1, yb)
        (ml, bl), (mr, br) = edges
        x = xx[ids]
        result[ids] += (
            _hinge_integral(mr, br - x, l, h)
            - _hinge_integral(ml, bl - x, l, h)
            - _hinge_integral(mr, br - x - 1, l, h)
            + _hinge_integral(ml, bl - x - 1, l, h)
        )
    result = np.clip(result, 0, 1)
    if pixel_indices is not None:
        return result
    out = np.zeros(shape, np.float64)
    out[yy, xx] = result
    return out


def geos_coverage(
    theta,
    dx=0.0,
    dy=0.0,
    shape=(256, 256),
    center=(128.0, 128.0),
    width=20.0,
    height=190.0,
):
    p = vertices(theta, center[0] + dx, center[1] + dy, width, height)
    poly = Polygon(p)
    lo = np.maximum(0, np.floor(p.min(0)).astype(int))
    hi = np.minimum(shape[::-1], np.ceil(p.max(0)).astype(int))
    yy, xx = np.mgrid[lo[1] : hi[1], lo[0] : hi[0]]
    cells = shapely.box(xx, yy, xx + 1, yy + 1)
    # GEOS robust overlay, double coordinate precision and no precision snapping.
    a = shapely.area(shapely.intersection(cells, poly, grid_size=0))
    out = np.zeros(shape, np.float64)
    out[yy, xx] = a
    return out


def srgb(y):
    return np.where(
        y <= 0.0031308, 12.92 * y, 1.055 * np.maximum(y, 0) ** (1 / 2.4) - 0.055
    )


def linear(q):
    return np.where(q <= 0.04045, q / 12.92, ((q + 0.055) / 1.055) ** 2.4)


def encode(a, bits=8):
    return np.rint(srgb(BG + (FG - BG) * a) * ((1 << bits) - 1)).astype(
        np.uint16 if bits > 8 else np.uint8
    )


def decode(q, bits=8):
    return np.clip(
        (linear(np.asarray(q, dtype=float) / ((1 << bits) - 1)) - BG) / (FG - BG), 0, 1
    )


def code_intervals(q, bits=8):
    n = (1 << bits) - 1
    return (
        np.clip(
            (linear(np.maximum(0, np.asarray(q, dtype=float) - 0.5) / n) - BG)
            / (FG - BG),
            0,
            1,
        ),
        np.clip(
            (linear(np.minimum(n, np.asarray(q, dtype=float) + 0.5) / n) - BG)
            / (FG - BG),
            0,
            1,
        ),
    )


def pca_axis(a):
    yy, xx = np.indices(a.shape, dtype=float)
    xx += 0.5
    yy += 0.5
    z = a.sum()
    cx = (a * xx).sum() / z
    cy = (a * yy).sum() / z
    x = xx - cx
    y = yy - cy
    v = np.linalg.eigh(
        [
            [np.sum(a * x * x) / z, np.sum(a * x * y) / z],
            [np.sum(a * x * y) / z, np.sum(a * y * y) / z],
        ]
    )[1][:, -1]
    if v[1] < 0:
        v = -v
    return math.degrees(math.atan2(v[0], v[1]))


def edge_axis(a, lower=None, upper=None):
    """Image-only straight-edge fit; never accepts nominal angle or translation.

    PCA chooses whether to transpose only. Rows from the central 60% of the
    nonzero support avoid both short caps for this elongated 20×190 stimulus.
    Each long-edge coordinate is an integral of local row coverage. Fit the two
    edge intercepts and their common slope. Optional code-derived intervals give
    LP bounds under the explicitly checked straight-edge/box-filter model.
    """
    init = pca_axis(a)
    transpose = abs(init) > 45
    if transpose:
        a = a.T
        if lower is not None:
            lower, upper = lower.T, upper.T
    support = np.flatnonzero(np.any(a > 1e-12, axis=1))
    if len(support) < 10:
        return {"status": "insufficient_rows"}
    lo, hi = support[0], support[-1] + 1
    low = lo + int(0.2 * (hi - lo))
    high = hi - int(0.2 * (hi - lo))
    positions = []
    design = []
    intervals = []
    invalid = 0
    for y in range(low, high):
        row = a[y]
        occupied = np.flatnonzero(row > 1e-12)
        full = np.flatnonzero(row > 0.98)
        if len(full) < 5 or len(occupied) < 5:
            invalid += 1
            continue
        # Three-pixel guards contain the <=1-pixel-per-row long-edge motion.
        j = max(0, occupied[0] - 2)
        k = min(full[0] + 2, int(np.mean(full)))
        j2 = max(full[-1] - 1, int(np.mean(full)))
        k2 = min(len(row), occupied[-1] + 3)
        for side, start, stop in [(0, j, k), (1, j2, k2)]:
            if stop <= start:
                invalid += 1
                continue
            if side == 0:
                position = stop - row[start:stop].sum()
                interval = (
                    None
                    if lower is None
                    else (
                        stop - upper[y, start:stop].sum(),
                        stop - lower[y, start:stop].sum(),
                    )
                )
            else:
                position = start + row[start:stop].sum()
                interval = (
                    None
                    if lower is None
                    else (
                        start + lower[y, start:stop].sum(),
                        start + upper[y, start:stop].sum(),
                    )
                )
            positions.append(position)
            design.append(
                [
                    y + 0.5 - (low + high) / 2,
                    1.0 if side == 0 else 0.0,
                    1.0 if side == 1 else 0.0,
                ]
            )
            if interval is not None:
                intervals.append(interval)
    X = np.asarray(design)
    z = np.asarray(positions)
    if len(z) < 10:
        return {"status": "insufficient_edges"}
    coef = np.linalg.lstsq(X, z, rcond=None)[0]
    m = coef[0]

    def axis(m):
        return (
            (
                math.degrees(math.atan2(1, m))
                if transpose
                else math.degrees(math.atan(m))
            )
            + 90
        ) % 180 - 90

    result = {
        "status": "ok",
        "axis_deg": axis(m),
        "initial_image_pca_deg": init,
        "transpose": transpose,
        "edge_rows": len(z) // 2,
        "edge_fit_rms_px": float(np.sqrt(np.mean((X @ coef - z) ** 2))),
        "edge_fit_max_abs_px": float(np.max(np.abs(X @ coef - z))),
        "skipped_rows": invalid,
    }
    if intervals:
        L, U = np.asarray(intervals).T
        A = np.vstack([X, -X])
        b = np.r_[U, -L]
        fits = [
            linprog(
                c,
                A_ub=A,
                b_ub=b,
                bounds=[(-1.01, 1.01), (None, None), (None, None)],
                method="highs",
            )
            for c in ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
        ]
        feasible = all(x.success for x in fits)
        result["quantization_interval_feasible"] = feasible
        if feasible:
            bounds = sorted([axis(x.x[0]) for x in fits])
            # Axis wraps at ±90; express local offsets around the image estimate.
            offsets = sorted(
                [((v - result["axis_deg"] + 90) % 180 - 90) for v in bounds]
            )
            result.update(
                {
                    "axis_interval_low_deg": result["axis_deg"] + offsets[0],
                    "axis_interval_high_deg": result["axis_deg"] + offsets[1],
                    "axis_interval_width_deg": offsets[1] - offsets[0],
                }
            )
        else:
            result["interval_failure"] = [x.message for x in fits]
    return result


def quantized_edge_axis(q, bits=8):
    """Image-only nonlinear half-plane fit plus exact code-bin feasibility.

    Each selected pixel constrains one straight edge. At fixed slope the box
    half-plane integral is analytically invertible, giving an interval for the
    intercept. Intersect all those intervals separately for the two long edges.
    The final angular feasible interval is conditional on this declared model;
    neither true theta nor true subpixel center is accepted by this function.
    """
    a = decode(q, bits)
    L, U = code_intervals(q, bits)
    initial = edge_axis(a, L, U)
    if initial.get("status") != "ok":
        return initial
    transpose = initial["transpose"]
    if transpose:
        a, L, U, q = a.T, L.T, U.T, q.T
    support = np.flatnonzero(np.any(a > 1e-12, axis=1))
    lo, hi = support[0], support[-1] + 1
    low = lo + int(0.2 * (hi - lo))
    high = hi - int(0.2 * (hi - lo))
    yc = (low + high) / 2
    xs = []
    ys = []
    sides = []
    obs = []
    lower = []
    upper = []
    for y in range(low, high):
        occupied = np.flatnonzero(a[y] > 1e-12)
        full = np.flatnonzero(a[y] > 0.98)
        if len(full) < 5:
            continue
        windows = [
            (0, max(0, occupied[0] - 2), full[0] + 2),
            (1, full[-1] - 1, min(a.shape[1], occupied[-1] + 3)),
        ]
        for side, start, stop in windows:
            for x in range(start, stop):
                xs.append(x)
                ys.append(y + 0.5 - yc)
                sides.append(side)
                obs.append(q[y, x] / ((1 << bits) - 1))
                lower.append(L[y, x])
                upper.append(U[y, x])
    x = np.asarray(xs)
    y = np.asarray(ys)
    side = np.asarray(sides)
    obs = np.asarray(obs)
    pl = np.asarray(lower)
    pu = np.asarray(upper)

    def coverage(par):
        m, b0, b1 = par
        z = m * y + np.where(side == 0, b0, b1)
        # Symmetric row integration of one half plane, evaluated with local
        # coordinates, avoids loss of precision from absolute image positions.
        t = np.where(side == 0, x + 1 - z, z - x)
        h = abs(m) / 2
        if h < 1e-9:
            return np.clip(t, 0, 1)
        v = 2 * h
        small = min(1.0, v)
        large = max(1.0, v)
        f = np.where(
            t <= -h,
            0,
            np.where(
                t < small - h,
                (t + h) ** 2 / (2 * v),
                np.where(
                    t <= large - h,
                    (t + (large - 1) / 2) / large,
                    np.where(t < 1 + h, 1 - (1 + h - t) ** 2 / (2 * v), 1),
                ),
            ),
        )
        return np.clip(f, 0, 1)

    ma = (
        math.tan(math.radians(initial["axis_deg"]))
        if not transpose
        else 1 / math.tan(math.radians(initial["axis_deg"]))
    )
    # Read both intercepts from the observed edge-coordinate fit, without GT.
    zs = []
    X = []
    for sid in (0, 1):
        ids = np.flatnonzero(
            (side == sid)
            & (decode(np.rint(obs * ((1 << bits) - 1)).astype(np.uint16), bits) > 0.01)
            & (decode(np.rint(obs * ((1 << bits) - 1)).astype(np.uint16), bits) < 0.99)
        )
        aa = decode(np.rint(obs * ((1 << bits) - 1)).astype(np.uint16), bits)
        pos = np.where(sid == 0, x + 1 - aa, x + aa)
        zs.append(
            float(np.median(pos[ids] - ma * y[ids]))
            if len(ids)
            else float(np.median(pos[side == sid] - ma * y[side == sid]))
        )
    fit = least_squares(
        lambda p: srgb(BG + (FG - BG) * coverage(p)) - obs,
        [ma, *zs],
        bounds=([-1.01, -np.inf, -np.inf], [1.01, np.inf, np.inf]),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=100,
    )

    def axis(m):
        return (
            (
                math.degrees(math.atan2(1, m))
                if transpose
                else math.degrees(math.atan(m))
            )
            + 90
        ) % 180 - 90

    def inverse_area(p, m):
        v = abs(m)
        if v < 1e-12:
            return p.copy()
        small = min(1.0, v)
        large = max(1.0, v)
        tail = small / (2 * large)
        return np.where(
            p < tail,
            np.sqrt(np.maximum(0, 2 * v * p)) - v / 2,
            np.where(
                p <= 1 - tail,
                large * p + (1 - large) / 2,
                1 + v / 2 - np.sqrt(np.maximum(0, 2 * v * (1 - p))),
            ),
        )

    def slack(m):
        invlo = inverse_area(pl, m)
        invhi = inverse_area(pu, m)
        bl = np.where(side == 0, x + 1 - invhi, x + invlo) - m * y
        bu = np.where(side == 0, x + 1 - invlo, x + invhi) - m * y
        bl = np.where(np.where(side == 0, pu >= 1, pl <= 0), -np.inf, bl)
        bu = np.where(np.where(side == 0, pl <= 0, pu >= 1), np.inf, bu)
        return max(float(bl[side == s].max() - bu[side == s].min()) for s in (0, 1))

    m = float(fit.x[0])
    result = {
        "status": "ok",
        "axis_deg": axis(m),
        "initial_image_edge_deg": initial["axis_deg"],
        "transpose": transpose,
        "pixels": len(x),
        "edge_rows": initial["edge_rows"],
        "sRGB_code_fit_rms": float(
            np.sqrt(np.mean((srgb(BG + (FG - BG) * coverage(fit.x)) - obs) ** 2))
            * ((1 << bits) - 1)
        ),
        "sRGB_code_fit_max_abs": float(
            np.max(abs(srgb(BG + (FG - BG) * coverage(fit.x)) - obs))
            * ((1 << bits) - 1)
        ),
        "optimizer_success": bool(fit.success),
        "quantization_interval_feasible": False,
    }
    # Outer bounds are the earlier conservative row-integral LP, independent of
    # this nonlinear refinement. At most ±.1 slope is searched if LP mismatches.
    if initial.get("quantization_interval_feasible"):
        aa = [initial["axis_interval_low_deg"], initial["axis_interval_high_deg"]]
        mm = sorted(
            [
                (
                    math.tan(math.radians(v))
                    if not transpose
                    else 1 / math.tan(math.radians(v))
                )
                for v in aa
            ]
        )
        ml, mh = max(-1.01, mm[0]), min(1.01, mm[1])
    else:
        ml, mh = max(-1.01, m - 0.1), min(1.01, m + 0.1)
    # Search a slope delta around the image-only fit. Bounded minimization on
    # an absolute slope near .5 otherwise stops at sqrt(machine_eps)*.5 even
    # when xatol is tighter; 16-bit feasible intervals can be narrower than it.
    opt = minimize_scalar(
        lambda d: slack(m + d),
        bounds=(ml - m, mh - m),
        method="bounded",
        options={"xatol": 1e-15, "maxiter": 200},
    )
    mid = m + float(opt.x)
    if slack(m) < slack(mid):
        mid = m
    result["minimum_intercept_interval_gap_px"] = float(slack(mid))
    if slack(mid) <= 1e-9:
        tol = 1e-10
        left = (
            ml
            if slack(ml) <= tol
            else brentq(lambda z: slack(z) - tol, ml, mid, xtol=1e-15)
        )
        right = (
            mh
            if slack(mh) <= tol
            else brentq(lambda z: slack(z) - tol, mid, mh, xtol=1e-15)
        )
        off = sorted(
            [((axis(v) - result["axis_deg"] + 90) % 180 - 90) for v in (left, right)]
        )
        result.update(
            {
                "quantization_interval_feasible": True,
                "axis_interval_low_deg": result["axis_deg"] + off[0],
                "axis_interval_high_deg": result["axis_deg"] + off[1],
                "axis_interval_width_deg": off[1] - off[0],
                "feasible_midpoint_axis_deg": axis((left + right) / 2),
            }
        )
    return result


def circle_pixel_area(x0, x1, y0, y1, radius):
    """Integral of max(0,min(y1,h)-max(y0,-h)), h=sqrt(r²-x²)."""
    lo, hi = max(x0, -radius), min(x1, radius)
    if lo >= hi:
        return 0.0
    cuts = [lo, hi]
    for y in (y0, y1):
        if abs(y) < radius:
            crossing = math.sqrt(max(0.0, radius * radius - y * y))
            cuts.extend(x for x in (-crossing, crossing) if lo < x < hi)
    cuts = sorted(set(cuts))

    def primitive(x):
        h = math.sqrt(max(0.0, radius * radius - x * x))
        return 0.5 * (x * h + radius * radius * math.asin(np.clip(x / radius, -1, 1)))

    area = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        h = math.sqrt(max(0.0, radius * radius - ((a + b) / 2) ** 2))
        if min(y1, h) <= max(y0, -h):
            continue
        top_const, top_h = (y1, 0) if y1 <= h else (0, 1)
        bot_const, bot_h = (y0, 0) if y0 >= -h else (0, -1)
        area += (top_const - bot_const) * (b - a) + (top_h - bot_h) * (
            primitive(b) - primitive(a)
        )
    return float(np.clip(area, 0, 1))


def disk_coverage(shape=(512, 768), center=(224.0, 302.0), area=5118.7529296875):
    radius = math.sqrt(area / math.pi)
    alpha = np.zeros(shape, np.float64)
    boundary_errors = []
    cx, cy = center
    for y in range(
        max(0, math.floor(cy - radius)), min(shape[0], math.ceil(cy + radius))
    ):
        for x in range(
            max(0, math.floor(cx - radius)), min(shape[1], math.ceil(cx + radius))
        ):
            dx, dy = max(abs(x - cx), abs(x + 1 - cx)), max(
                abs(y - cy), abs(y + 1 - cy)
            )
            if dx * dx + dy * dy <= radius * radius:
                alpha[y, x] = 1.0
                continue
            x0, x1, y0, y1 = x - cx, x + 1 - cx, y - cy, y + 1 - cy
            value = circle_pixel_area(x0, x1, y0, y1, radius)
            alpha[y, x] = value
            if 1e-12 < value < 1 - 1e-12:
                lo, hi = max(x0, -radius), min(x1, radius)
                cuts = []
                for edge in (y0, y1):
                    if abs(edge) < radius:
                        q = math.sqrt(radius * radius - edge * edge)
                        cuts.extend(v for v in (-q, q) if lo < v < hi)

                def integrand(v):
                    h = math.sqrt(max(0.0, radius * radius - v * v))
                    return max(0.0, min(y1, h) - max(y0, -h))

                independent, _ = quad(
                    integrand,
                    lo,
                    hi,
                    points=sorted(set(cuts)),
                    epsabs=1e-11,
                    epsrel=1e-11,
                )
                boundary_errors.append(abs(value - independent))
    area_error = abs(float(alpha.sum()) - area)
    assert area_error < 1e-8, area_error
    assert max(boundary_errors, default=0) < 1e-8
    return alpha, {
        "radius_px": radius,
        "diameter_px": 2 * radius,
        "target_area_px2": area,
        "actual_alpha_area_px2": float(alpha.sum()),
        "area_absolute_error_px2": area_error,
        "independent_boundary_pixels": len(boundary_errors),
        "quadrature_max_coverage_error": max(boundary_errors, default=0),
    }
