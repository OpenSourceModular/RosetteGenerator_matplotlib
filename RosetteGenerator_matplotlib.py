import json
import math
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib.pyplot as plt
import numpy as np


ROSETTE_TYPES = ["Bump", "Dip", "Arch", "Concave+Convex", "Puffy", "W", "X + 1", "Flat", "Lotus", "A", "Sine"]
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


def get_rosette_geometry(kind, radius, count, height, extra=None):
    if radius <= 0:
        raise ValueError("Radius must be greater than 0")
    if count < 1:
        raise ValueError("Count must be at least 1")

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
    else:
        raise ValueError("Rosette style is not implemented for drawing")

    return segments, reference_radius, title, reference_label


def export_curve_only_svg(
    path,
    kind,
    radius,
    count,
    height,
    extra=None,
    stroke=CURVE_COLOR,
    stroke_width=0.25,
):
    segments, _, _, _ = get_rosette_geometry(kind, radius, count, height, extra)

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


def draw_rosette(kind, radius, count, height, extra=None, show=True, curve_only=False):
    segments, reference_radius, title, reference_label = get_rosette_geometry(
        kind, radius, count, height, extra
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.canvas.manager.set_window_title(title)

    if not curve_only:
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

    max_extent = radius * 1.15
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_aspect("equal", adjustable="box")
    if curve_only:
        ax.axis("off")
        fig.patch.set_alpha(0.0)
    else:
        ax.grid(True, linestyle=":", linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.legend(loc="upper right")
    if show:
        plt.show()
    return fig


class RosetteGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rosette Generator using Matplotlib")

        self.option_var = tk.StringVar(value=ROSETTE_TYPES[0])
        self.field_vars = {}
        self.last_rosette_config = None
        self.defaults = self._load_defaults()

        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.grid(row=0, column=0, sticky="nsew")

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

        self.dynamic_frame = ttk.Frame(self.main_frame)
        self.dynamic_frame.grid(row=1, column=0, sticky="ew")

        button_row = ttk.Frame(self.main_frame)
        button_row.grid(row=2, column=0, sticky="e", pady=(10, 0))

        create_btn = ttk.Button(button_row, text="Create", command=self.on_create)
        create_btn.pack(side="left", padx=(0, 6))

        export_btn = ttk.Button(button_row, text="Export SVG", command=self.on_export_svg)
        export_btn.pack(side="left", padx=(0, 6))

        close_btn = ttk.Button(button_row, text="Close", command=self.root.destroy)
        close_btn.pack(side="left", padx=(0, 6))

        defaults_btn = ttk.Button(button_row, text="Defaults", command=self.on_defaults)
        defaults_btn.pack(side="left")

        self.on_option_changed()

    def clear_dynamic_fields(self):
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()
        self.field_vars = {}

    def add_field(self, row, label, default=""):
        ttk.Label(self.dynamic_frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=default)
        entry = ttk.Entry(self.dynamic_frame, textvariable=var, width=18)
        entry.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=2)
        self.field_vars[label] = var

    def _defaults_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "rosette_defaults.json")

    def _load_defaults(self):
        path = self._defaults_path()
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {
                    "outer_radius": float(data.get("outer_radius", 50)),
                    "amplitude": float(data.get("amplitude", 5)),
                    "num_segments": int(data.get("num_segments", 12)),
                }
            except Exception:
                pass
        return {"outer_radius": 50.0, "amplitude": 5.0, "num_segments": 12}

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

        def on_save():
            try:
                new_radius = float(radius_var.get())
                new_amplitude = float(amplitude_var.get())
                new_segments = int(segments_var.get())
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numbers.", parent=dialog)
                return
            self.defaults["outer_radius"] = new_radius
            self.defaults["amplitude"] = new_amplitude
            self.defaults["num_segments"] = new_segments
            path = self._defaults_path()
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.defaults, f, indent=2)
            except Exception as exc:
                messagebox.showerror("Save Failed", str(exc), parent=dialog)
                return
            dialog.destroy()

        btn_row = ttk.Frame(frame)
        btn_row.grid(row=3, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(btn_row, text="Save", command=on_save).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Cancel", command=dialog.destroy).pack(side="left")

    def on_option_changed(self, _event=None):
        selected = self.option_var.get()
        self.clear_dynamic_fields()
        r = str(self.defaults["outer_radius"])
        a = str(self.defaults["amplitude"])
        n = str(self.defaults["num_segments"])

        if selected == "Bump":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Number of Segments", default=n)
            self.add_field(2, "Amplitude", default=a)
        elif selected == "Dip":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Number of Segments", default=n)
            self.add_field(2, "Amplitude", default=a)
        elif selected == "Arch":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Number of Segments", default=n)
            self.add_field(2, "Amplitude", default=a)
        elif selected == "Concave+Convex":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Number of Segments", default=n)
            self.add_field(2, "Amplitude", default=a)
            self.add_field(3, "Split %", default="50")
        elif selected == "Puffy":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Number of Segments", default=n)
            self.add_field(2, "Amplitude", default=a)
        elif selected == "W":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Number of Segments", default=n)
            self.add_field(2, "Amplitude", default=a)
        elif selected == "X + 1":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Amplitude", default=a)
            self.add_field(2, "Number of Segments", default=n)
            self.add_field(3, "X")
        elif selected == "Flat":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Number of Segments", default=n)
        elif selected == "Lotus":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Number of Segments", default=n)
            self.add_field(2, "Amplitude", default=a)
        elif selected == "A":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Number of Segments", default=n)
            self.add_field(2, "Amplitude", default=a)
        elif selected == "Sine":
            self.add_field(0, "Outer Radius", default=r)
            self.add_field(1, "Amplitude", default=a)
            self.add_field(2, "Number of Segments", default=n)

    def _get_selected_parameters(self):
        selected = self.option_var.get()
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
            }
        raise ValueError("Rosette style is not implemented for drawing")

    def on_create(self):
        try:
            config = self._get_selected_parameters()
            draw_rosette(
                config["kind"],
                config["radius"],
                config["count"],
                config["height"],
                extra=config["extra"],
                show=True,
            )
            self.last_rosette_config = config
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc))
        except Exception as exc:
            messagebox.showerror("Create Failed", str(exc))

    def on_export_svg(self):
        if self.last_rosette_config is None:
            messagebox.showinfo("Export SVG", "Create a rosette first.")
            return

        kind = self.last_rosette_config["kind"]
        radius = self.last_rosette_config["radius"]
        count = self.last_rosette_config["count"]
        height = self.last_rosette_config["height"]
        extra = self.last_rosette_config["extra"]
        default_name = "{0}-{1}.svg".format(kind, count)
        path = filedialog.asksaveasfilename(
            title="Export Rosette as SVG",
            defaultextension=".svg",
            initialfile=default_name,
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            export_curve_only_svg(
                path,
                kind,
                radius,
                count,
                height,
                extra=extra,
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
