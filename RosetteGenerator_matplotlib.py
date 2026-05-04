import json
import math
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
    from shapely.ops import unary_union
except ImportError:
    GeometryCollection = None
    MultiPolygon = None
    Polygon = None
    unary_union = None


ROSETTE_TYPES = ["Bump", "Dip", "Arch", "Concave+Convex", "Puffy", "W", "X + 1", "Flat", "Lotus", "A", "Sine", "Bead"]
TAU = 2.0 * math.pi
CURVE_COLOR = "#000000"


def _normalize_angle(angle):
    return angle % TAU


def _is_between_ccw(start, target, end):
    span = (_normalize_angle(end) - _normalize_angle(start)) % TAU
    reach = (_normalize_angle(target) - _normalize_angle(start)) % TAU
    return reach <= span + 1e-12


def _distance(p0, p1):
    return math.hypot(p0[0] - p1[0], p0[1] - p1[1])


def _rotate_point(point, angle_rad):
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    x, y = point
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)


def _rotate_segments(segments, angle_rad):
    if abs(angle_rad) <= 1e-12:
        return segments

    rotated = []
    for segment in segments:
        if segment[0] == "arc":
            _, p0, p1, p2 = segment
            rotated.append(
                (
                    "arc",
                    _rotate_point(p0, angle_rad),
                    _rotate_point(p1, angle_rad),
                    _rotate_point(p2, angle_rad),
                )
            )
        else:
            _, p0, p1 = segment
            rotated.append(("line", _rotate_point(p0, angle_rad), _rotate_point(p1, angle_rad)))
    return rotated


def arc_through_three_points(p0, p1, p2, samples=60):
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = p2

    det = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(det) < 1e-12:
        raise ValueError("Arc points are collinear. Increase rosette height or reduce count.")

    x1_sq_y1_sq = x1 * x1 + y1 * y1
    x2_sq_y2_sq = x2 * x2 + y2 * y2
    x3_sq_y3_sq = x3 * x3 + y3 * y3

    cx = (
        x1_sq_y1_sq * (y2 - y3)
        + x2_sq_y2_sq * (y3 - y1)
        + x3_sq_y3_sq * (y1 - y2)
    ) / det
    cy = (
        x1_sq_y1_sq * (x3 - x2)
        + x2_sq_y2_sq * (x1 - x3)
        + x3_sq_y3_sq * (x2 - x1)
    ) / det

    radius = math.hypot(x1 - cx, y1 - cy)

    a0 = math.atan2(y1 - cy, x1 - cx)
    am = math.atan2(y2 - cy, x2 - cx)
    a2 = math.atan2(y3 - cy, x3 - cx)

    if _is_between_ccw(a0, am, a2):
        span = (_normalize_angle(a2) - _normalize_angle(a0)) % TAU
        angles = np.linspace(a0, a0 + span, samples)
    else:
        span = (_normalize_angle(a0) - _normalize_angle(a2)) % TAU
        angles = np.linspace(a0, a0 - span, samples)

    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    return x, y


def generate_bump_arcs(radius, count, height):
    ref_radius = radius - height
    if ref_radius <= 0:
        raise ValueError("Height of Bumps must be smaller than Radius")

    arcs = []
    angle_step = TAU / float(count)
    half_span = angle_step / 2.0

    for i in range(count):
        center = i * angle_step
        start = center - half_span
        end = center + half_span

        p_start = (ref_radius * math.cos(start), ref_radius * math.sin(start))
        p_peak = (radius * math.cos(center), radius * math.sin(center))
        p_end = (ref_radius * math.cos(end), ref_radius * math.sin(end))

        arcs.append((p_start, p_peak, p_end))

    return arcs, ref_radius


def generate_dip_arcs(radius, count, height):
    inner_radius = radius - height
    if inner_radius <= 0:
        raise ValueError("Height of Dips must be smaller than Radius")

    arcs = []
    angle_step = TAU / float(count)
    half_span = angle_step / 2.0

    for i in range(count):
        center = i * angle_step
        start = center - half_span
        end = center + half_span

        p_start = (radius * math.cos(start), radius * math.sin(start))
        p_dip = (inner_radius * math.cos(center), inner_radius * math.sin(center))
        p_end = (radius * math.cos(end), radius * math.sin(end))

        arcs.append((p_start, p_dip, p_end))

    return arcs, inner_radius


def generate_concave_convex_arcs(radius, count, height, split_pct=0.5):
    r_mid = radius - height / 2.0
    r_inner = radius - height
    if r_inner <= 0:
        raise ValueError("Height must be smaller than Radius")
    if not (0.0 < split_pct < 1.0):
        raise ValueError("Split % must be between 0 and 100 (exclusive)")

    segments = []
    angle_step = TAU / float(count)
    half_span = angle_step / 2.0

    convex_span = (1.0 - split_pct) * angle_step
    concave_span = split_pct * angle_step

    for i in range(count):
        center = i * angle_step
        start = center - half_span
        split_angle = start + convex_span

        p_bump_start = (r_mid * math.cos(start), r_mid * math.sin(start))
        p_bump_peak = (radius * math.cos(start + convex_span / 2.0), radius * math.sin(start + convex_span / 2.0))
        p_bump_end = (r_mid * math.cos(split_angle), r_mid * math.sin(split_angle))

        p_dip_start = p_bump_end
        p_dip_valley = (r_inner * math.cos(split_angle + concave_span / 2.0), r_inner * math.sin(split_angle + concave_span / 2.0))
        p_dip_end = (r_mid * math.cos(start + angle_step), r_mid * math.sin(start + angle_step))

        segments.append(("arc", p_bump_start, p_bump_peak, p_bump_end))
        segments.append(("arc", p_dip_start, p_dip_valley, p_dip_end))

    return segments, r_mid


def _line_to_inner_at_45(outer_point, outer_angle, inner_radius, toward_ccw):
    radial = (math.cos(outer_angle), math.sin(outer_angle))
    tangent = (-math.sin(outer_angle), math.cos(outer_angle))
    tangent_sign = 1.0 if toward_ccw else -1.0

    # 45-degree inward line in local polar frame: equal inward and tangential components.
    root_two = math.sqrt(2.0)
    direction = (
        (tangent_sign * tangent[0] - radial[0]) / root_two,
        (tangent_sign * tangent[1] - radial[1]) / root_two,
    )

    x0, y0 = outer_point
    dx, dy = direction

    b = 2.0 * (x0 * dx + y0 * dy)
    c = (x0 * x0 + y0 * y0) - (inner_radius * inner_radius)
    disc = (b * b) - (4.0 * c)
    if disc <= 0.0:
        raise ValueError(
            "Height of Arches is too large for 45-degree side lines at this radius."
        )

    sqrt_disc = math.sqrt(disc)
    lam_1 = (-b - sqrt_disc) / 2.0
    lam_2 = (-b + sqrt_disc) / 2.0
    positive_solutions = [value for value in (lam_1, lam_2) if value > 1e-12]
    if not positive_solutions:
        raise ValueError("Could not construct 45-degree arch side line.")

    lam = min(positive_solutions)
    return (x0 + lam * dx, y0 + lam * dy)


def generate_arch_segments(radius, count, height):
    inner_radius = radius - height
    if inner_radius <= 0:
        raise ValueError("Height of Arches must be smaller than Radius")

    segments = []
    angle_step = TAU / float(count)
    half_span = angle_step / 2.0

    for i in range(count):
        center = i * angle_step
        start = center - half_span
        end = center + half_span

        p_outer_start = (radius * math.cos(start), radius * math.sin(start))
        p_outer_peak = (radius * math.cos(center), radius * math.sin(center))
        p_outer_end = (radius * math.cos(end), radius * math.sin(end))

        p_inner_start = _line_to_inner_at_45(
            p_outer_start, start, inner_radius, toward_ccw=True
        )
        p_inner_end = _line_to_inner_at_45(
            p_outer_end, end, inner_radius, toward_ccw=False
        )

        segments.append(("line", p_outer_start, p_inner_start))
        segments.append(("arc", p_inner_start, p_outer_peak, p_inner_end))
        segments.append(("line", p_inner_end, p_outer_end))

    return segments, inner_radius


def generate_puffy_segments(radius, count, offset):
    if offset < 0:
        raise ValueError("Offset must be greater than or equal to 0")

    segments = []
    angle_step = TAU / float(count)
    half_span = angle_step / 2.0

    for i in range(count):
        center_angle = i * angle_step
        start_angle = center_angle - half_span
        end_angle = center_angle + half_span

        p_start = (radius * math.cos(start_angle), radius * math.sin(start_angle))
        p_end = (radius * math.cos(end_angle), radius * math.sin(end_angle))

        arc_center = (
            offset * math.cos(center_angle + math.pi),
            offset * math.sin(center_angle + math.pi),
        )
        arc_radius = _distance(arc_center, p_start)
        p_mid = (
            arc_center[0] + arc_radius * math.cos(center_angle),
            arc_center[1] + arc_radius * math.sin(center_angle),
        )

        segments.append(("arc", p_start, p_mid, p_end))

    return segments, radius


def generate_w_segments(radius, count, height):
    inner_radius = radius - height
    if inner_radius <= 0:
        raise ValueError("Height must be smaller than Radius")

    segments = []
    angle_step = TAU / float(count)
    half_span = angle_step / 2.0

    for i in range(count):
        center_angle = i * angle_step
        start_angle = center_angle - half_span
        end_angle = center_angle + half_span

        p_start = (radius * math.cos(start_angle), radius * math.sin(start_angle))
        p_mid = (
            inner_radius * math.cos(center_angle),
            inner_radius * math.sin(center_angle),
        )
        p_end = (radius * math.cos(end_angle), radius * math.sin(end_angle))

        segments.append(("line", p_start, p_mid))
        segments.append(("line", p_mid, p_end))

    return segments, inner_radius


def generate_x_plus_one_segments(radius, count, height, x_count):
    inner_radius = radius - height
    if inner_radius <= 0:
        raise ValueError("Height must be smaller than Radius")
    if x_count < 1:
        raise ValueError("X must be at least 1")

    segments = []
    angle_step = TAU / float(count)
    half_span = angle_step / 2.0

    for i in range(count):
        segment_start = i * angle_step
        midpoint = segment_start + half_span

        first_peak_angle = segment_start + (half_span / 2.0)
        p_first_start = (
            inner_radius * math.cos(segment_start),
            inner_radius * math.sin(segment_start),
        )
        p_first_peak = (
            radius * math.cos(first_peak_angle),
            radius * math.sin(first_peak_angle),
        )
        p_first_end = (
            inner_radius * math.cos(midpoint),
            inner_radius * math.sin(midpoint),
        )
        segments.append(("arc", p_first_start, p_first_peak, p_first_end))

        sub_span = half_span / float(x_count)
        for j in range(x_count):
            sub_start_angle = midpoint + (j * sub_span)
            sub_end_angle = sub_start_angle + sub_span
            sub_peak_angle = sub_start_angle + (sub_span / 2.0)

            p_sub_start = (
                inner_radius * math.cos(sub_start_angle),
                inner_radius * math.sin(sub_start_angle),
            )
            p_sub_peak = (
                radius * math.cos(sub_peak_angle),
                radius * math.sin(sub_peak_angle),
            )
            p_sub_end = (
                inner_radius * math.cos(sub_end_angle),
                inner_radius * math.sin(sub_end_angle),
            )
            segments.append(("arc", p_sub_start, p_sub_peak, p_sub_end))

    return segments, inner_radius


def generate_flat_segments(radius, count):
    segments = []
    angle_step = TAU / float(count)

    for i in range(count):
        start_angle = i * angle_step
        end_angle = ((i + 1) % count) * angle_step

        p_start = (radius * math.cos(start_angle), radius * math.sin(start_angle))
        p_end = (radius * math.cos(end_angle), radius * math.sin(end_angle))
        segments.append(("line", p_start, p_end))

    return segments, radius


def generate_lotus_segments(radius, count, height):
    inner_radius = radius - height
    if inner_radius <= 0:
        raise ValueError("Height must be smaller than Radius")

    saddle_radius = radius - (0.35 * height)
    segments = []
    angle_step = TAU / float(count)
    half_span = angle_step / 2.0
    quarter_span = angle_step / 4.0

    for i in range(count):
        center_angle = i * angle_step
        start_angle = center_angle - half_span
        end_angle = center_angle + half_span

        p_start = (inner_radius * math.cos(start_angle), inner_radius * math.sin(start_angle))
        p_saddle = (
            saddle_radius * math.cos(center_angle),
            saddle_radius * math.sin(center_angle),
        )
        p_end = (inner_radius * math.cos(end_angle), inner_radius * math.sin(end_angle))

        p_left_peak = (
            radius * math.cos(center_angle - quarter_span),
            radius * math.sin(center_angle - quarter_span),
        )
        p_right_peak = (
            radius * math.cos(center_angle + quarter_span),
            radius * math.sin(center_angle + quarter_span),
        )

        segments.append(("arc", p_start, p_left_peak, p_saddle))
        segments.append(("arc", p_saddle, p_right_peak, p_end))

    return segments, saddle_radius


def generate_a_segments(radius, count, height):
    inner_radius = radius - height
    if inner_radius <= 0:
        raise ValueError("Height must be smaller than Radius")

    transition_radius = radius - (0.35 * height)
    segments = []
    angle_step = TAU / float(count)
    third_span = angle_step / 3.0

    for i in range(count):
        start_angle = i * angle_step
        first_split_angle = start_angle + third_span
        second_split_angle = start_angle + (2.0 * third_span)
        end_angle = start_angle + angle_step

        p_start = (radius * math.cos(start_angle), radius * math.sin(start_angle))
        p_first_split = (
            inner_radius * math.cos(first_split_angle),
            inner_radius * math.sin(first_split_angle),
        )
        p_second_split = (
            inner_radius * math.cos(second_split_angle),
            inner_radius * math.sin(second_split_angle),
        )
        p_end = (radius * math.cos(end_angle), radius * math.sin(end_angle))

        p_first_ctrl = (
            transition_radius * math.cos(start_angle + (third_span / 2.0)),
            transition_radius * math.sin(start_angle + (third_span / 2.0)),
        )
        p_center_peak = (
            radius * math.cos(start_angle + (angle_step / 2.0)),
            radius * math.sin(start_angle + (angle_step / 2.0)),
        )
        p_last_ctrl = (
            transition_radius * math.cos(second_split_angle + (third_span / 2.0)),
            transition_radius * math.sin(second_split_angle + (third_span / 2.0)),
        )

        segments.append(("arc", p_start, p_first_ctrl, p_first_split))
        segments.append(("arc", p_first_split, p_center_peak, p_second_split))
        segments.append(("arc", p_second_split, p_last_ctrl, p_end))

    return segments, inner_radius


def generate_sine_segments(radius, count, amplitude, samples_per_period=120):
    inner_radius = radius - amplitude
    if amplitude <= 0:
        raise ValueError("Amplitude must be greater than 0")
    if inner_radius <= 0:
        raise ValueError("Amplitude must be smaller than Radius")

    total_samples = max(count * samples_per_period, 2)
    theta_values = np.linspace(0.0, TAU, total_samples + 1)
    radial_values = (radius - (amplitude / 2.0)) + ((amplitude / 2.0) * np.sin((count * theta_values) + (math.pi / 2.0)))

    segments = []
    points = [
        (radial * math.cos(theta), radial * math.sin(theta))
        for theta, radial in zip(theta_values, radial_values)
    ]

    for start_point, end_point in zip(points, points[1:]):
        segments.append(("line", start_point, end_point))

    return segments, radius - (amplitude / 2.0)


def generate_bead_segments(radius, count, amplitude, flat_length):
    if amplitude <= 0:
        raise ValueError("Amplitude must be greater than 0")
    if flat_length < 0:
        raise ValueError("Flat length must be greater than or equal to 0")

    construction_radius = radius - (amplitude / 2.0)
    inner_radius = radius - amplitude
    if construction_radius <= 0 or inner_radius <= 0:
        raise ValueError("Amplitude must be smaller than Radius")

    angle_step = TAU / float(count)
    flat_angle = flat_length / construction_radius
    if (2.0 * flat_angle) >= angle_step:
        raise ValueError("Flat length is too large for the selected radius and segment count")

    bulge_span = (angle_step - (2.0 * flat_angle)) / 2.0
    if bulge_span <= 1e-6:
        raise ValueError("Flat length leaves no room for bead bulges")

    segments = []

    for i in range(count):
        segment_start = i * angle_step
        first_arc_end = segment_start + bulge_span
        first_flat_end = first_arc_end + flat_angle
        second_arc_end = first_flat_end + bulge_span
        segment_end = segment_start + angle_step

        p_first_start = (
            construction_radius * math.cos(segment_start),
            construction_radius * math.sin(segment_start),
        )
        p_first_peak = (
            radius * math.cos(segment_start + (bulge_span / 2.0)),
            radius * math.sin(segment_start + (bulge_span / 2.0)),
        )
        p_first_end = (
            construction_radius * math.cos(first_arc_end),
            construction_radius * math.sin(first_arc_end),
        )
        segments.append(("arc", p_first_start, p_first_peak, p_first_end))

        if flat_angle > 1e-6:
            p_first_flat_mid = (
                construction_radius * math.cos(first_arc_end + (flat_angle / 2.0)),
                construction_radius * math.sin(first_arc_end + (flat_angle / 2.0)),
            )
            p_first_flat_end = (
                construction_radius * math.cos(first_flat_end),
                construction_radius * math.sin(first_flat_end),
            )
            segments.append(("arc", p_first_end, p_first_flat_mid, p_first_flat_end))
        else:
            p_first_flat_end = p_first_end

        p_second_valley = (
            inner_radius * math.cos(first_flat_end + (bulge_span / 2.0)),
            inner_radius * math.sin(first_flat_end + (bulge_span / 2.0)),
        )
        p_second_arc_end = (
            construction_radius * math.cos(second_arc_end),
            construction_radius * math.sin(second_arc_end),
        )
        segments.append(("arc", p_first_flat_end, p_second_valley, p_second_arc_end))

        if flat_angle > 1e-6:
            p_second_flat_mid = (
                construction_radius * math.cos(second_arc_end + (flat_angle / 2.0)),
                construction_radius * math.sin(second_arc_end + (flat_angle / 2.0)),
            )
            p_second_flat_end = (
                construction_radius * math.cos(segment_end),
                construction_radius * math.sin(segment_end),
            )
            segments.append(("arc", p_second_arc_end, p_second_flat_mid, p_second_flat_end))

    return segments, construction_radius


def get_rosette_geometry(kind, radius, count, height, extra=None, phase=0.0):
    if radius <= 0:
        raise ValueError("Radius must be greater than 0")
    if count < 1:
        raise ValueError("Count must be at least 1")
    if phase < 0.0 or phase > 180.0:
        raise ValueError("Phase must be between 0 and 180 degrees")

    if kind == "Bump":
        if height <= 0:
            raise ValueError("Height must be greater than 0")
        arc_triplets, reference_radius = generate_bump_arcs(radius, count, height)
        segments = [("arc", p0, p1, p2) for (p0, p1, p2) in arc_triplets]
        title = "Bump-{0}".format(count)
        reference_label = "Reference Circle (Radius - Height)"
    elif kind == "Dip":
        if height <= 0:
            raise ValueError("Height must be greater than 0")
        arc_triplets, reference_radius = generate_dip_arcs(radius, count, height)
        segments = [("arc", p0, p1, p2) for (p0, p1, p2) in arc_triplets]
        title = "Dip-{0}".format(count)
        reference_label = "Reference Circle (Radius - Height)"
    elif kind == "Arch":
        if height <= 0:
            raise ValueError("Height must be greater than 0")
        segments, reference_radius = generate_arch_segments(radius, count, height)
        title = "Arch-{0}".format(count)
        reference_label = "Reference Circle (Radius - Height)"
    elif kind == "Concave+Convex":
        if height <= 0:
            raise ValueError("Height must be greater than 0")
        split_pct = extra if extra is not None else 0.5
        segments, reference_radius = generate_concave_convex_arcs(radius, count, height, split_pct)
        title = "Concave+Convex-{0}".format(count)
        reference_label = "Mid Circle (Radius - Height/2)"
    elif kind == "Puffy":
        segments, reference_radius = generate_puffy_segments(radius, count, height)
        title = "Puffy-{0}".format(count)
        reference_label = "Outer Radius"
    elif kind == "W":
        if height <= 0:
            raise ValueError("Height must be greater than 0")
        segments, reference_radius = generate_w_segments(radius, count, height)
        title = "W-{0}".format(count)
        reference_label = "Reference Circle (Radius - Height)"
    elif kind == "X + 1":
        if height <= 0:
            raise ValueError("Height must be greater than 0")
        if extra is None:
            raise ValueError("X value is required for X + 1 style")
        segments, reference_radius = generate_x_plus_one_segments(
            radius, count, height, extra
        )
        title = "X + 1-{0}".format(count)
        reference_label = "Reference Circle (Radius - Height)"
    elif kind == "Flat":
        segments, reference_radius = generate_flat_segments(radius, count)
        title = "Flat-{0}".format(count)
        reference_label = "Outer Radius"
    elif kind == "Lotus":
        if height <= 0:
            raise ValueError("Height must be greater than 0")
        segments, reference_radius = generate_lotus_segments(radius, count, height)
        title = "Lotus-{0}".format(count)
        reference_label = "Saddle Circle"
    elif kind == "A":
        if height <= 0:
            raise ValueError("Height must be greater than 0")
        segments, reference_radius = generate_a_segments(radius, count, height)
        title = "A-{0}".format(count)
        reference_label = "Reference Circle (Radius - Height)"
    elif kind == "Sine":
        segments, reference_radius = generate_sine_segments(radius, count, height)
        title = "Sine-{0}".format(count)
        reference_label = "Mid Radius"
    elif kind == "Bead":
        if extra is None:
            raise ValueError("Flat length is required for Bead style")
        segments, reference_radius = generate_bead_segments(radius, count, height, extra)
        title = "Bead-{0}".format(count)
        reference_label = "Construction Circle (Radius - Amplitude/2)"
    else:
        raise ValueError("Rosette style is not implemented for drawing")

    # Positive phase rotates to the right (clockwise).
    segments = _rotate_segments(segments, -math.radians(phase))

    return segments, reference_radius, title, reference_label


def export_curve_only_svg(
    path,
    kind,
    radius,
    count,
    height,
    extra=None,
    phase=0.0,
    stroke=CURVE_COLOR,
    stroke_width=0.25,
):
    segments, _, _, _ = get_rosette_geometry(kind, radius, count, height, extra, phase=phase)

    curve_points = []
    all_x = []
    all_y = []
    for segment in segments:
        if segment[0] == "arc":
            _, p0, p1, p2 = segment
            x, y = arc_through_three_points(p0, p1, p2)
        else:
            _, p0, p1 = segment
            x = np.array([p0[0], p1[0]], dtype=float)
            y = np.array([p0[1], p1[1]], dtype=float)

        curve_points.append((x, y))
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())

    if not all_x or not all_y:
        raise ValueError("No curve data was generated for export")

    min_x = min(all_x)
    max_x = max(all_x)
    min_y = min(all_y)
    max_y = max(all_y)
    margin = max(stroke_width * 2.0, 0.5)

    view_min_x = min_x - margin
    view_min_y = min_y - margin
    view_w = (max_x - min_x) + (2.0 * margin)
    view_h = (max_y - min_y) + (2.0 * margin)

    svg_lines = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>",
        "<svg",
        "   version=\"1.1\"",
        "   viewBox=\"{0:.6f} {1:.6f} {2:.6f} {3:.6f}\"".format(
            view_min_x, view_min_y, view_w, view_h
        ),
        "   id=\"svg1\"",
        "   sodipodi:docname=\"{0}\"".format(os.path.basename(path)),
        "   inkscape:version=\"1.4.2\"",
        "   xmlns:inkscape=\"http://www.inkscape.org/namespaces/inkscape\"",
        "   xmlns:sodipodi=\"http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd\"",
        "   xmlns=\"http://www.w3.org/2000/svg\"",
        "   xmlns:svg=\"http://www.w3.org/2000/svg\">",
        "  <defs id=\"defs1\" />",
        "  <sodipodi:namedview",
        "     id=\"namedview1\"",
        "     pagecolor=\"#ffffff\"",
        "     bordercolor=\"#000000\"",
        "     borderopacity=\"0.25\"",
        "     inkscape:showpageshadow=\"2\"",
        "     inkscape:pageopacity=\"0.0\"",
        "     inkscape:pagecheckerboard=\"0\"",
        "     inkscape:deskcolor=\"#d1d1d1\"",
        "     inkscape:current-layer=\"svg1\" />",
    ]

    path_parts = []
    if curve_points:
        x_first, y_first = curve_points[0]
        path_parts.append("M {0:.6f} {1:.6f}".format(x_first[0], y_first[0]))
        for idx, (x, y) in enumerate(curve_points):
            start_index = 1 if idx == 0 else 0
            for px, py in zip(x[start_index:], y[start_index:]):
                path_parts.append("L {0:.6f} {1:.6f}".format(px, py))
        path_parts.append("Z")

    if path_parts:
        d_value = " ".join(path_parts)
        svg_lines.append(
            "  <path d=\"{0}\" fill=\"none\" stroke=\"{1}\" stroke-width=\"{2:.6f}\" stroke-linecap=\"round\" stroke-linejoin=\"round\" id=\"path1\" />".format(
                d_value, stroke, stroke_width
            )
        )

    svg_lines.append("</svg>")

    with open(path, "w", encoding="utf-8") as out_file:
        out_file.write("\n".join(svg_lines))


def export_geometry_svg(
    path,
    geometry,
    stroke=CURVE_COLOR,
    stroke_width=0.25,
):
    if geometry is None or geometry.is_empty:
        raise ValueError("No merged geometry was generated for export")

    min_x, min_y, max_x, max_y = geometry.bounds
    margin = max(stroke_width * 2.0, 0.5)

    view_min_x = min_x - margin
    view_min_y = min_y - margin
    view_w = (max_x - min_x) + (2.0 * margin)
    view_h = (max_y - min_y) + (2.0 * margin)

    svg_lines = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>",
        "<svg",
        "   version=\"1.1\"",
        "   viewBox=\"{0:.6f} {1:.6f} {2:.6f} {3:.6f}\"".format(
            view_min_x, view_min_y, view_w, view_h
        ),
        "   id=\"svg1\"",
        "   sodipodi:docname=\"{0}\"".format(os.path.basename(path)),
        "   inkscape:version=\"1.4.2\"",
        "   xmlns:inkscape=\"http://www.inkscape.org/namespaces/inkscape\"",
        "   xmlns:sodipodi=\"http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd\"",
        "   xmlns=\"http://www.w3.org/2000/svg\"",
        "   xmlns:svg=\"http://www.w3.org/2000/svg\">",
        "  <defs id=\"defs1\" />",
        "  <sodipodi:namedview",
        "     id=\"namedview1\"",
        "     pagecolor=\"#ffffff\"",
        "     bordercolor=\"#000000\"",
        "     borderopacity=\"0.25\"",
        "     inkscape:showpageshadow=\"2\"",
        "     inkscape:pageopacity=\"0.0\"",
        "     inkscape:pagecheckerboard=\"0\"",
        "     inkscape:deskcolor=\"#d1d1d1\"",
        "     inkscape:current-layer=\"svg1\" />",
    ]

    path_parts = []
    for polygon in _iter_polygon_parts(geometry):
        exterior_points = list(polygon.exterior.coords)
        if not exterior_points:
            continue
        path_parts.append(
            "M {0:.6f} {1:.6f}".format(exterior_points[0][0], exterior_points[0][1])
        )
        for px, py in exterior_points[1:]:
            path_parts.append("L {0:.6f} {1:.6f}".format(px, py))
        path_parts.append("Z")

        for interior in polygon.interiors:
            interior_points = list(interior.coords)
            if not interior_points:
                continue
            path_parts.append(
                "M {0:.6f} {1:.6f}".format(interior_points[0][0], interior_points[0][1])
            )
            for px, py in interior_points[1:]:
                path_parts.append("L {0:.6f} {1:.6f}".format(px, py))
            path_parts.append("Z")

    if path_parts:
        d_value = " ".join(path_parts)
        svg_lines.append(
            "  <path d=\"{0}\" fill=\"none\" stroke=\"{1}\" stroke-width=\"{2:.6f}\" stroke-linecap=\"round\" stroke-linejoin=\"round\" id=\"path1\" />".format(
                d_value, stroke, stroke_width
            )
        )

    svg_lines.append("</svg>")

    with open(path, "w", encoding="utf-8") as out_file:
        out_file.write("\n".join(svg_lines))


def draw_rosette(kind, radius, count, height, extra=None, phase=0.0, show=True, curve_only=False):
    segments, reference_radius, title, reference_label = get_rosette_geometry(
        kind, radius, count, height, extra, phase=phase
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.canvas.manager.set_window_title(title)
    _draw_rosette_on_axes(
        ax,
        segments,
        reference_radius,
        radius,
        title,
        reference_label,
        curve_only=curve_only,
    )
    if show:
        plt.show()
    return fig


def _draw_rosette_on_axes(
    ax,
    segments,
    reference_radius,
    radius,
    title,
    reference_label,
    clear_axes=True,
    include_reference=True,
    show_title=True,
    show_legend=True,
    view_radius=None,
    polar_grid=False,
    curve_only=False,
):
    if clear_axes:
        ax.clear()

    if not curve_only and include_reference:
        theta = np.linspace(0.0, TAU, 400)
        ax.plot(
            reference_radius * np.cos(theta),
            reference_radius * np.sin(theta),
            linestyle="--",
            linewidth=1.0,
            color="#808080",
            label=reference_label,
        )

    for segment in segments:
        if segment[0] == "arc":
            _, p0, p1, p2 = segment
            x, y = arc_through_three_points(p0, p1, p2)
        else:
            _, p0, p1 = segment
            x = np.array([p0[0], p1[0]], dtype=float)
            y = np.array([p0[1], p1[1]], dtype=float)
        ax.plot(x, y, color=CURVE_COLOR, linewidth=1.8)

    axis_radius = view_radius if view_radius is not None else radius
    max_extent = axis_radius * 1.15
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_aspect("equal", adjustable="box")

    if curve_only:
        ax.axis("off")
        ax.figure.patch.set_alpha(0.0)
    else:
        _apply_grid_mode(ax, axis_radius, polar_grid)
        if show_title:
            ax.set_title(title)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper right")


def _render_rosette_in_axes(
    ax,
    kind,
    radius,
    count,
    height,
    extra=None,
    phase=0.0,
    clear_axes=True,
    include_reference=True,
    show_title=True,
    show_legend=True,
    view_radius=None,
    polar_grid=False,
    curve_only=False,
):
    segments, reference_radius, title, reference_label = get_rosette_geometry(
        kind, radius, count, height, extra, phase=phase
    )
    _draw_rosette_on_axes(
        ax,
        segments,
        reference_radius,
        radius,
        title,
        reference_label,
        clear_axes=clear_axes,
        include_reference=include_reference,
        show_title=show_title,
        show_legend=show_legend,
        view_radius=view_radius,
        polar_grid=polar_grid,
        curve_only=curve_only,
    )


def _apply_grid_mode(ax, axis_radius, polar_grid):
    if not polar_grid:
        ax.grid(True, linestyle=":", linewidth=0.6)
        return

    ax.grid(False)
    grid_color = "#b0b0b0"

    theta = np.linspace(0.0, TAU, 361)
    ring_count = 4
    for idx in range(1, ring_count + 1):
        ring_radius = axis_radius * (idx / float(ring_count))
        ax.plot(
            ring_radius * np.cos(theta),
            ring_radius * np.sin(theta),
            linestyle=":",
            linewidth=0.6,
            color=grid_color,
            zorder=0,
        )

    for degrees in range(0, 360, 45):
        angle = math.radians(degrees)
        x = axis_radius * math.cos(angle)
        y = axis_radius * math.sin(angle)
        ax.plot(
            [-x, x],
            [-y, y],
            linestyle=":",
            linewidth=0.8,
            color=grid_color,
            zorder=0,
        )


def _segments_to_outline_points(segments):
    outline_points = []

    for segment in segments:
        if segment[0] == "arc":
            _, p0, p1, p2 = segment
            x, y = arc_through_three_points(p0, p1, p2, samples=120)
        else:
            _, p0, p1 = segment
            x = np.array([p0[0], p1[0]], dtype=float)
            y = np.array([p0[1], p1[1]], dtype=float)

        points = list(zip(x.tolist(), y.tolist()))
        if outline_points and points:
            points = points[1:]
        outline_points.extend(points)

    if outline_points and outline_points[0] != outline_points[-1]:
        outline_points.append(outline_points[0])

    return outline_points


def _build_rosette_geometry(kind, radius, count, height, extra=None, phase=0.0):
    if Polygon is None:
        raise RuntimeError("Merge requires the shapely package.")

    segments, _, title, _ = get_rosette_geometry(kind, radius, count, height, extra, phase=phase)
    outline_points = _segments_to_outline_points(segments)
    if len(outline_points) < 4:
        raise ValueError("Not enough points to build rosette geometry.")

    geometry = Polygon(outline_points)
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty:
        raise ValueError("Generated rosette geometry is empty.")

    return geometry, title


def _iter_polygon_parts(geometry):
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
        return
    if geometry.geom_type == "MultiPolygon":
        for polygon in geometry.geoms:
            yield polygon
        return
    if geometry.geom_type == "GeometryCollection":
        for part in geometry.geoms:
            yield from _iter_polygon_parts(part)


def _draw_geometry_on_axes(
    ax,
    geometry,
    title,
    clear_axes=True,
    show_title=True,
    view_radius=None,
    polar_grid=False,
):
    if clear_axes:
        ax.clear()

    for polygon in _iter_polygon_parts(geometry):
        x, y = polygon.exterior.xy
        ax.plot(x, y, color=CURVE_COLOR, linewidth=1.8)
        for interior in polygon.interiors:
            x, y = interior.xy
            ax.plot(x, y, color=CURVE_COLOR, linewidth=1.4)

    if view_radius is None:
        min_x, min_y, max_x, max_y = geometry.bounds
        axis_radius = max(abs(min_x), abs(max_x), abs(min_y), abs(max_y), 1.0)
    else:
        axis_radius = view_radius

    max_extent = axis_radius * 1.15
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_aspect("equal", adjustable="box")
    _apply_grid_mode(ax, axis_radius, polar_grid)
    if show_title:
        ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")


def _copy_draw_state(state):
    if state is None:
        return None
    if state["type"] == "config":
        return {"type": "config", "config": dict(state["config"])}
    return {"type": "geometry", "geometry": state["geometry"], "title": state["title"]}


def _draw_states_equal(left, right):
    if left is None or right is None:
        return False
    if left["type"] != right["type"]:
        return False
    if left["type"] == "config":
        return left["config"] == right["config"]
    return left["geometry"].equals(right["geometry"])


def _state_view_radius(state):
    if state is None:
        return 1.0
    if state["type"] == "config":
        return state["config"]["radius"]

    min_x, min_y, max_x, max_y = state["geometry"].bounds
    return max(abs(min_x), abs(max_x), abs(min_y), abs(max_y), 1.0)


class RosetteGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rosette Generator using Matplotlib")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.option_var = tk.StringVar(value=ROSETTE_TYPES[0])
        self.auto_draw_var = tk.BooleanVar(value=False)
        self.polar_grid_var = tk.BooleanVar(value=False)
        self.field_vars = {}
        self.last_rosette_config = None
        self.last_drawn_state = None
        self.held_rosette_config = None
        self.defaults = self._load_defaults()
        self.auto_draw_var.set(bool(self.defaults.get("auto_draw", False)))
        self.polar_grid_var.set(bool(self.defaults.get("polar_grid", False)))

        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(3, weight=1)

        option_row = ttk.Frame(self.main_frame)
        option_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(option_row, text="Option:").pack(side="left")
        self.option_combo = ttk.Combobox(
            option_row,
            state="readonly",
            values=ROSETTE_TYPES,
            textvariable=self.option_var,
            width=50,
        )
        self.option_combo.pack(side="left", padx=(8, 0))
        self.option_combo.bind("<<ComboboxSelected>>", self.on_option_changed)

        auto_draw_check = ttk.Checkbutton(
            option_row,
            text="Auto draw",
            variable=self.auto_draw_var,
            command=self._auto_draw_if_enabled,
        )
        auto_draw_check.pack(side="left", padx=(12, 0))

        polar_cartesian_check = ttk.Checkbutton(
            option_row,
            text="Polar/Cartesian",
            variable=self.polar_grid_var,
            command=self.on_grid_mode_changed,
        )
        polar_cartesian_check.pack(side="left", padx=(12, 0))

        self.dynamic_frame = ttk.Frame(self.main_frame)
        self.dynamic_frame.grid(row=1, column=0, sticky="ew")
        self.dynamic_frame.columnconfigure(1, weight=1)

        button_row = ttk.Frame(self.main_frame)
        button_row.grid(row=2, column=0, sticky="e", pady=(10, 0))

        create_btn = ttk.Button(button_row, text="Create", command=self.on_create)
        create_btn.pack(side="left", padx=(0, 6))

        export_btn = ttk.Button(button_row, text="Export SVG", command=self.on_export_svg)
        export_btn.pack(side="left", padx=(0, 6))

        clear_btn = ttk.Button(button_row, text="Clear", command=self.on_clear_graph)
        clear_btn.pack(side="left", padx=(0, 6))

        self.hold_btn = tk.Button(button_row, text="Hold", command=self.on_hold)
        self.hold_btn.pack(side="left", padx=(0, 6))
        self.default_hold_button_bg = self.hold_btn.cget("background")

        reset_btn = ttk.Button(button_row, text="Reset", command=self.on_reset)
        reset_btn.pack(side="left", padx=(0, 6))

        merge_btn = ttk.Button(button_row, text="Merge", command=self.on_merge)
        merge_btn.pack(side="left", padx=(0, 6))

        close_btn = ttk.Button(button_row, text="Close", command=self.root.destroy)
        close_btn.pack(side="left", padx=(0, 6))

        defaults_btn = ttk.Button(button_row, text="Defaults", command=self.on_defaults)
        defaults_btn.pack(side="left")

        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(7, 7), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.on_clear_graph()

        self.on_option_changed()

    def clear_dynamic_fields(self):
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()
        self.field_vars = {}

    def add_field(self, row, label, default=""):
        ttk.Label(self.dynamic_frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
        slider_specs = {
            "Outer Radius": {"type": "float", "from": 1.0, "to": 175.0, "resolution": 0.5},
            "Number of Segments": {"type": "int", "from": 1, "to": 96, "resolution": 1},
            "Amplitude": {"type": "float", "from": 0.0, "to": 50.0, "resolution": 0.5},
            "Flat Length": {"type": "float", "from": 0.0, "to": 80.0, "resolution": 0.5},
            "Phase": {"type": "float", "from": 0.0, "to": 180.0, "resolution": 0.5, "decimals": 2},
        }

        if label in slider_specs:
            spec = slider_specs[label]
            if spec["type"] == "int":
                try:
                    numeric_default = int(float(default))
                except ValueError:
                    numeric_default = spec["from"]
                numeric_default = max(spec["from"], min(spec["to"], numeric_default))
                var = tk.IntVar(value=numeric_default)
            else:
                try:
                    numeric_default = float(default)
                except ValueError:
                    numeric_default = spec["from"]
                numeric_default = max(spec["from"], min(spec["to"], numeric_default))
                var = tk.DoubleVar(value=numeric_default)

            var.trace_add("write", self.on_field_changed)
            slider = tk.Scale(
                self.dynamic_frame,
                from_=spec["from"],
                to=spec["to"],
                orient="horizontal",
                resolution=spec["resolution"],
                showvalue=True,
                variable=var,
                length=260,
            )
            slider.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=2)

            entry_var = tk.StringVar()

            def _format_value(value):
                if spec["type"] == "int":
                    return str(int(round(float(value))))
                if "decimals" in spec:
                    decimals = int(spec["decimals"])
                else:
                    resolution = float(spec["resolution"])
                    decimals = max(1, len(str(resolution).split(".")[-1].rstrip("0")))
                return ("{0:." + str(decimals) + "f}").format(float(value))

            def _sync_entry_from_slider(*_args):
                entry_var.set(_format_value(var.get()))

            def _apply_entry_value(_event=None):
                text = entry_var.get().strip()
                if not text:
                    entry_var.set(_format_value(var.get()))
                    return
                try:
                    raw_value = float(text)
                except ValueError:
                    entry_var.set(_format_value(var.get()))
                    return

                min_value = float(spec["from"])
                max_value = float(spec["to"])
                clamped = max(min_value, min(max_value, raw_value))
                if spec["type"] == "int":
                    var.set(int(round(clamped)))
                else:
                    var.set(clamped)
                entry_var.set(_format_value(var.get()))

            _sync_entry_from_slider()
            var.trace_add("write", _sync_entry_from_slider)
            entry = ttk.Entry(self.dynamic_frame, textvariable=entry_var, width=8, justify="right")
            entry.grid(row=row, column=2, sticky="w", padx=(8, 0), pady=2)
            entry.bind("<Return>", _apply_entry_value)
            entry.bind("<FocusOut>", _apply_entry_value)
        else:
            var = tk.StringVar(value=default)
            var.trace_add("write", self.on_field_changed)
            entry = ttk.Entry(self.dynamic_frame, textvariable=var, width=18)
            entry.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=2)

        self.field_vars[label] = var

    def on_field_changed(self, *_args):
        self._auto_draw_if_enabled()

    def _defaults_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "rosette_defaults.json")

    def _load_defaults(self):
        def _parse_bool(value, fallback=False):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in ("1", "true", "yes", "on"):
                    return True
                if normalized in ("0", "false", "no", "off"):
                    return False
            if isinstance(value, (int, float)):
                return bool(value)
            return fallback

        path = self._defaults_path()
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                default_save_dir = str(data.get("default_save_dir", "")).strip()
                if default_save_dir and not os.path.isdir(default_save_dir):
                    default_save_dir = ""
                return {
                    "outer_radius": float(data.get("outer_radius", 50)),
                    "amplitude": float(data.get("amplitude", 5)),
                    "num_segments": int(data.get("num_segments", 12)),
                    "auto_draw": _parse_bool(data.get("auto_draw", False), False),
                    "polar_grid": _parse_bool(data.get("polar_grid", False), False),
                    "default_save_dir": default_save_dir,
                }
            except Exception:
                pass
        return {
            "outer_radius": 50.0,
            "amplitude": 5.0,
            "num_segments": 12,
            "auto_draw": False,
            "polar_grid": False,
            "default_save_dir": "",
        }

    def on_defaults(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Defaults")
        dialog.resizable(False, False)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="Outer Radius:").grid(row=0, column=0, sticky="w", pady=4)
        radius_var = tk.StringVar(value=str(self.defaults["outer_radius"]))
        ttk.Entry(frame, textvariable=radius_var, width=14).grid(row=0, column=1, padx=(8, 0), pady=4)

        ttk.Label(frame, text="Amplitude:").grid(row=1, column=0, sticky="w", pady=4)
        amplitude_var = tk.StringVar(value=str(self.defaults["amplitude"]))
        ttk.Entry(frame, textvariable=amplitude_var, width=14).grid(row=1, column=1, padx=(8, 0), pady=4)

        ttk.Label(frame, text="Number of Segments:").grid(row=2, column=0, sticky="w", pady=4)
        segments_var = tk.StringVar(value=str(self.defaults["num_segments"]))
        ttk.Entry(frame, textvariable=segments_var, width=14).grid(row=2, column=1, padx=(8, 0), pady=4)

        auto_draw_default_var = tk.BooleanVar(value=bool(self.defaults.get("auto_draw", False)))
        ttk.Checkbutton(
            frame,
            text="Auto draw",
            variable=auto_draw_default_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=4)

        polar_grid_default_var = tk.BooleanVar(value=bool(self.defaults.get("polar_grid", False)))
        ttk.Checkbutton(
            frame,
            text="Polar/Cartesian",
            variable=polar_grid_default_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Label(frame, text="Default Save Directory:").grid(row=5, column=0, sticky="w", pady=4)
        save_dir_var = tk.StringVar(value=self.defaults.get("default_save_dir", ""))
        save_dir_entry = ttk.Entry(frame, textvariable=save_dir_var, width=32)
        save_dir_entry.grid(row=5, column=1, padx=(8, 0), pady=4, sticky="ew")

        def on_browse_save_dir():
            chosen_dir = filedialog.askdirectory(
                title="Select Default Save Directory",
                mustexist=True,
                parent=dialog,
            )
            if chosen_dir:
                save_dir_var.set(chosen_dir)

        ttk.Button(frame, text="Browse...", command=on_browse_save_dir).grid(
            row=5, column=2, padx=(8, 0), pady=4, sticky="w"
        )

        def on_save():
            try:
                new_radius = float(radius_var.get())
                new_amplitude = float(amplitude_var.get())
                new_segments = int(segments_var.get())
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numbers.", parent=dialog)
                return
            default_save_dir = save_dir_var.get().strip()
            if default_save_dir and not os.path.isdir(default_save_dir):
                messagebox.showerror(
                    "Invalid Directory",
                    "Default Save Directory must be an existing folder.",
                    parent=dialog,
                )
                return
            self.defaults["outer_radius"] = new_radius
            self.defaults["amplitude"] = new_amplitude
            self.defaults["num_segments"] = new_segments
            self.defaults["auto_draw"] = bool(auto_draw_default_var.get())
            self.defaults["polar_grid"] = bool(polar_grid_default_var.get())
            self.defaults["default_save_dir"] = default_save_dir
            path = self._defaults_path()
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.defaults, f, indent=2)
            except Exception as exc:
                messagebox.showerror("Save Failed", str(exc), parent=dialog)
                return

            self.auto_draw_var.set(self.defaults["auto_draw"])
            self.polar_grid_var.set(self.defaults["polar_grid"])
            self.on_grid_mode_changed()
            dialog.destroy()

        btn_row = ttk.Frame(frame)
        btn_row.grid(row=6, column=0, columnspan=3, sticky="e", pady=(10, 0))
        ttk.Button(btn_row, text="Save", command=on_save).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Cancel", command=dialog.destroy).pack(side="left")

    def on_option_changed(self, _event=None):
        selected = self.option_var.get()
        self.clear_dynamic_fields()
        r = str(self.defaults["outer_radius"])
        a = str(self.defaults["amplitude"])
        n = str(self.defaults["num_segments"])
        p = "0"

        if selected == "Bump":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Number of Segments", default=n)
            self.add_field(2, "Amplitude", default=a)
            self.add_field(3, "Phase", default=p)
        elif selected == "Dip":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Number of Segments", default=n)
            self.add_field(2, "Amplitude", default=a)
            self.add_field(3, "Phase", default=p)
        elif selected == "Arch":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Number of Segments", default=n)
            self.add_field(2, "Amplitude", default=a)
            self.add_field(3, "Phase", default=p)
        elif selected == "Concave+Convex":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Number of Segments", default=n)
            self.add_field(2, "Amplitude", default=a)
            self.add_field(3, "Split %", default="50")
            self.add_field(4, "Phase", default=p)
        elif selected == "Puffy":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Number of Segments", default=n)
            self.add_field(2, "Amplitude", default=a)
            self.add_field(3, "Phase", default=p)
        elif selected == "W":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Number of Segments", default=n)
            self.add_field(2, "Amplitude", default=a)
            self.add_field(3, "Phase", default=p)
        elif selected == "X + 1":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Amplitude", default=a)
            self.add_field(2, "Number of Segments", default=n)
            self.add_field(3, "X", default="3")
            self.add_field(4, "Phase", default=p)
        elif selected == "Flat":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Number of Segments", default=n)
            self.add_field(2, "Phase", default=p)
        elif selected == "Lotus":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Number of Segments", default=n)
            self.add_field(2, "Amplitude", default=a)
            self.add_field(3, "Phase", default=p)
        elif selected == "A":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Number of Segments", default=n)
            self.add_field(2, "Amplitude", default=a)
            self.add_field(3, "Phase", default=p)
        elif selected == "Sine":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Amplitude", default=a)
            self.add_field(2, "Number of Segments", default=n)
            self.add_field(3, "Phase", default=p)
        elif selected == "Bead":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Amplitude", default=a)
            self.add_field(2, "Number of Segments", default=n)
            self.add_field(3, "Flat Length", default="8.0")
            self.add_field(4, "Phase", default=p)

        self._auto_draw_if_enabled()

    def _get_selected_parameters(self):
        selected = self.option_var.get()
        phase = float(self.field_vars["Phase"].get()) if "Phase" in self.field_vars else 0.0
        if selected == "Bump":
            radius = float(self.field_vars["Outer Radius"].get())
            count = int(self.field_vars["Number of Segments"].get())
            height = float(self.field_vars["Amplitude"].get())
            return {
                "kind": selected,
                "radius": radius,
                "count": count,
                "height": height,
                "extra": None,
                "phase": phase,
            }
        if selected == "Dip":
            radius = float(self.field_vars["Outer Radius"].get())
            count = int(self.field_vars["Number of Segments"].get())
            height = float(self.field_vars["Amplitude"].get())
            return {
                "kind": selected,
                "radius": radius,
                "count": count,
                "height": height,
                "extra": None,
                "phase": phase,
            }
        if selected == "Arch":
            radius = float(self.field_vars["Outer Radius"].get())
            count = int(self.field_vars["Number of Segments"].get())
            height = float(self.field_vars["Amplitude"].get())
            return {
                "kind": selected,
                "radius": radius,
                "count": count,
                "height": height,
                "extra": None,
                "phase": phase,
            }
        if selected == "Concave+Convex":
            radius = float(self.field_vars["Outer Radius"].get())
            count = int(self.field_vars["Number of Segments"].get())
            height = float(self.field_vars["Amplitude"].get())
            split_pct = float(self.field_vars["Split %"].get().strip().rstrip("%")) / 100.0
            if not (0.0 < split_pct < 1.0):
                raise ValueError("Split % must be between 0 and 100 (exclusive)")
            return {
                "kind": selected,
                "radius": radius,
                "count": count,
                "height": height,
                "extra": split_pct,
                "phase": phase,
            }
        if selected == "Puffy":
            radius = float(self.field_vars["Outer Radius"].get())
            count = int(self.field_vars["Number of Segments"].get())
            offset = float(self.field_vars["Amplitude"].get())
            return {
                "kind": selected,
                "radius": radius,
                "count": count,
                "height": offset,
                "extra": None,
                "phase": phase,
            }
        if selected == "W":
            radius = float(self.field_vars["Outer Radius"].get())
            count = int(self.field_vars["Number of Segments"].get())
            height = float(self.field_vars["Amplitude"].get())
            return {
                "kind": selected,
                "radius": radius,
                "count": count,
                "height": height,
                "extra": None,
                "phase": phase,
            }
        if selected == "X + 1":
            radius = float(self.field_vars["Outer Radius"].get())
            height = float(self.field_vars["Amplitude"].get())
            count = int(self.field_vars["Number of Segments"].get())
            x_count = int(self.field_vars["X"].get())
            return {
                "kind": selected,
                "radius": radius,
                "count": count,
                "height": height,
                "extra": x_count,
                "phase": phase,
            }
        if selected == "Flat":
            radius = float(self.field_vars["Outer Radius"].get())
            count = int(self.field_vars["Number of Segments"].get())
            return {
                "kind": selected,
                "radius": radius,
                "count": count,
                "height": 0.0,
                "extra": None,
                "phase": phase,
            }
        if selected == "Lotus":
            radius = float(self.field_vars["Outer Radius"].get())
            count = int(self.field_vars["Number of Segments"].get())
            height = float(self.field_vars["Amplitude"].get())
            return {
                "kind": selected,
                "radius": radius,
                "count": count,
                "height": height,
                "extra": None,
                "phase": phase,
            }
        if selected == "A":
            radius = float(self.field_vars["Outer Radius"].get())
            count = int(self.field_vars["Number of Segments"].get())
            height = float(self.field_vars["Amplitude"].get())
            return {
                "kind": selected,
                "radius": radius,
                "count": count,
                "height": height,
                "extra": None,
                "phase": phase,
            }
        if selected == "Sine":
            radius = float(self.field_vars["Outer Radius"].get())
            amplitude = float(self.field_vars["Amplitude"].get())
            count = int(self.field_vars["Number of Segments"].get())
            return {
                "kind": selected,
                "radius": radius,
                "count": count,
                "height": amplitude,
                "extra": None,
                "phase": phase,
            }
        if selected == "Bead":
            radius = float(self.field_vars["Outer Radius"].get())
            amplitude = float(self.field_vars["Amplitude"].get())
            count = int(self.field_vars["Number of Segments"].get())
            flat_length = float(self.field_vars["Flat Length"].get())
            return {
                "kind": selected,
                "radius": radius,
                "count": count,
                "height": amplitude,
                "extra": flat_length,
                "phase": phase,
            }
        raise ValueError("Rosette style is not implemented for drawing")

    def on_create(self):
        self._draw_current_config(show_errors=True, clear_first=True)

    def _draw_current_config(self, show_errors, clear_first):
        try:
            config = self._get_selected_parameters()
            current_state = {"type": "config", "config": dict(config)}
            held_state = self.held_rosette_config
            has_distinct_held = held_state is not None and not _draw_states_equal(held_state, current_state)
            view_radius = max(_state_view_radius(current_state), _state_view_radius(held_state))

            if clear_first:
                self.ax.clear()
                self.figure.patch.set_alpha(1.0)

            if has_distinct_held:
                self._render_state(
                    held_state,
                    clear_axes=False,
                    show_title=False,
                    show_legend=False,
                    include_reference=False,
                    view_radius=view_radius,
                )

            self._render_state(
                current_state,
                clear_axes=False if has_distinct_held else True,
                show_title=True,
                show_legend=True,
                include_reference=True,
                view_radius=view_radius,
            )
            self.canvas.draw_idle()
            self.last_rosette_config = config
            self.last_drawn_state = current_state
        except ValueError as exc:
            if show_errors:
                messagebox.showerror("Invalid Input", str(exc))
        except Exception as exc:
            if show_errors:
                messagebox.showerror("Create Failed", str(exc))

    def _render_state(
        self,
        state,
        clear_axes,
        show_title,
        show_legend=False,
        include_reference=False,
        view_radius=None,
    ):
        if state["type"] == "config":
            config = state["config"]
            _render_rosette_in_axes(
                self.ax,
                config["kind"],
                config["radius"],
                config["count"],
                config["height"],
                extra=config["extra"],
                phase=config.get("phase", 0.0),
                clear_axes=clear_axes,
                include_reference=include_reference,
                show_title=show_title,
                show_legend=show_legend,
                view_radius=view_radius,
                polar_grid=self.polar_grid_var.get(),
            )
            return

        _draw_geometry_on_axes(
            self.ax,
            state["geometry"],
            state["title"],
            clear_axes=clear_axes,
            show_title=show_title,
            view_radius=view_radius,
            polar_grid=self.polar_grid_var.get(),
        )

    def on_grid_mode_changed(self):
        if self.last_drawn_state is None:
            return

        if self.last_drawn_state["type"] == "config":
            self._draw_current_config(show_errors=False, clear_first=True)
            return

        self._render_state(
            self.last_drawn_state,
            clear_axes=True,
            show_title=True,
            view_radius=_state_view_radius(self.last_drawn_state),
        )
        self.figure.patch.set_alpha(1.0)
        self.canvas.draw_idle()

    def _auto_draw_if_enabled(self):
        if self.auto_draw_var.get():
            self._draw_current_config(show_errors=False, clear_first=True)

    def on_clear_graph(self):
        self.ax.clear()
        self.ax.axis("off")
        self.ax.text(
            0.5,
            0.5,
            "Graph cleared",
            ha="center",
            va="center",
            transform=self.ax.transAxes,
            color="#606060",
        )
        self.figure.patch.set_alpha(1.0)
        self.canvas.draw_idle()
        self.last_rosette_config = None
        self.last_drawn_state = None

    def on_hold(self):
        if self.last_drawn_state is None:
            messagebox.showinfo("Hold", "Create a rosette first.")
            return
        self.held_rosette_config = _copy_draw_state(self.last_drawn_state)
        self.hold_btn.configure(background="yellow")

    def on_reset(self):
        self.held_rosette_config = None
        self.hold_btn.configure(background=self.default_hold_button_bg)
        self.on_clear_graph()

    def on_merge(self):
        if unary_union is None or Polygon is None:
            messagebox.showerror("Merge Unavailable", "Merge requires the shapely package.")
            return
        if self.held_rosette_config is None:
            messagebox.showinfo("Merge", "Hold a rosette first.")
            return
        if self.last_drawn_state is None:
            messagebox.showinfo("Merge", "Create a rosette first.")
            return

        try:
            held_geometry = self._state_to_geometry(self.held_rosette_config)
            current_geometry = self._state_to_geometry(self.last_drawn_state)
            merged_geometry = unary_union([held_geometry, current_geometry])
            if not merged_geometry.is_valid:
                merged_geometry = merged_geometry.buffer(0)
            if merged_geometry.is_empty:
                raise ValueError("Merged geometry is empty.")

            merged_state = {
                "type": "geometry",
                "geometry": merged_geometry,
                "title": "Merged Rosette",
            }
            view_radius = max(_state_view_radius(self.held_rosette_config), _state_view_radius(merged_state))
            self._render_state(
                merged_state,
                clear_axes=True,
                show_title=True,
                view_radius=view_radius,
            )
            self.figure.patch.set_alpha(1.0)
            self.canvas.draw_idle()
            self.last_drawn_state = merged_state
            self.last_rosette_config = None
        except ValueError as exc:
            messagebox.showerror("Merge Failed", str(exc))
        except Exception as exc:
            messagebox.showerror("Merge Failed", str(exc))

    def _state_to_geometry(self, state):
        if state["type"] == "geometry":
            return state["geometry"]

        config = state["config"]
        geometry, _ = _build_rosette_geometry(
            config["kind"],
            config["radius"],
            config["count"],
            config["height"],
            extra=config["extra"],
            phase=config.get("phase", 0.0),
        )
        return geometry

    def on_export_svg(self):
        if self.last_drawn_state is None:
            messagebox.showinfo("Export SVG", "Create a rosette first.")
            return

        if self.last_drawn_state["type"] == "geometry":
            default_name = "Merged-Rosette.svg"
        else:
            kind = self.last_drawn_state["config"]["kind"]
            count = self.last_drawn_state["config"]["count"]
            default_name = "{0}-{1}.svg".format(kind, count)

        path = filedialog.asksaveasfilename(
            title="Export Rosette as SVG",
            defaultextension=".svg",
            initialfile=default_name,
            initialdir=self.defaults.get("default_save_dir") or os.path.dirname(os.path.abspath(__file__)),
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            if self.last_drawn_state["type"] == "geometry":
                export_geometry_svg(path, self.last_drawn_state["geometry"])
            else:
                config = self.last_drawn_state["config"]
                export_curve_only_svg(
                    path,
                    config["kind"],
                    config["radius"],
                    config["count"],
                    config["height"],
                    extra=config["extra"],
                    phase=config.get("phase", 0.0),
                )
            messagebox.showinfo("Export Complete", "Saved SVG to:\n{0}".format(path))
        except Exception as exc:
            messagebox.showerror("Export Failed", str(exc))


def main():
    root = tk.Tk()
    app = RosetteGeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
