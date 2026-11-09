"""Learned normalized-layout detector.

Annotated objects usually sit in a stable layout around a reference shape.
Instead of learning a pixel classifier and sliding it over the image -- which is
slow, needs thousands of patches and easily returns the wrong number of boxes --
this detector learns *where* the annotated objects are relative to an anchor:

    centre_x = anchor_x + dx * anchor_r
    centre_y = anchor_y + dy * anchor_r
    width    = w  * anchor_r
    height   = h  * anchor_r

The anchor (centre + radius) is found with a Hough-circle search.  On frames
where the reference shape is not visible the search falls back to the anchor
learned during training, and the layout still produces exactly K boxes.  That is
the property this app needs: K annotated objects in, K detections out,
regardless of image contrast.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import cv2
import numpy as np

# The reference point occupies roughly these normalized coordinates.  They are
# the default fixed anchor and also bootstrap the Hough search; after training
# the detected anchor is stored in the model metadata.
DEFAULT_RADIUS_RATIO = 0.50
DEFAULT_CENTER = (0.52, 0.51)

DEFAULT_DETECTOR_CONFIG: dict[str, Any] = {
    "anchor_mode": "fixed",       # "hough" = find a circle, "fixed" = use center/radius
    "center_x": DEFAULT_CENTER[0],
    "center_y": DEFAULT_CENTER[1],
    "radius": DEFAULT_RADIUS_RATIO,
    "radius_min": 0.42,
    "radius_max": 0.61,
    "center_tolerance": 0.32,
    "radius_error": 0.16,
    "center_error": 0.28,
}


def normalize_detector_config(raw: Any) -> dict[str, Any]:
    """Clamp a user-supplied detector configuration to safe values.

    The defaults describe a centred round reference object, but every value can
    be changed from the UI for a different camera/object setup.  Keeping this in
    one place means training and prediction always use the same settings.
    """
    cfg = dict(DEFAULT_DETECTOR_CONFIG)
    if isinstance(raw, dict):
        mode = str(raw.get("anchor_mode", cfg["anchor_mode"])).lower()
        if mode in ("hough", "circle", "auto"):
            cfg["anchor_mode"] = "hough"
        elif mode in ("fixed", "manual", "none"):
            cfg["anchor_mode"] = "fixed"

        def number(key: str, low: float, high: float) -> None:
            try:
                value = float(raw.get(key, cfg[key]))
            except (TypeError, ValueError):
                return
            cfg[key] = max(low, min(high, value))

        number("center_x", 0.0, 1.0)
        number("center_y", 0.0, 1.0)
        number("radius", 0.02, 2.0)
        number("radius_min", 0.02, 2.0)
        number("radius_max", 0.02, 2.0)
        number("center_tolerance", 0.02, 2.0)
        number("radius_error", 0.02, 1.0)
        number("center_error", 0.02, 2.0)

    if cfg["radius_min"] > cfg["radius_max"]:
        cfg["radius_min"], cfg["radius_max"] = cfg["radius_max"], cfg["radius_min"]
    # Make sure the fallback radius is inside the searched range.
    cfg["radius_min"] = min(cfg["radius_min"], cfg["radius"])
    cfg["radius_max"] = max(cfg["radius_max"], cfg["radius"] + 0.02)
    return cfg

# Several Hough thresholds are tried, from strict to permissive.  The strict
# ones give clean circles on high-contrast frames, the permissive ones still
# find the anchor on dim frames.
HOUGH_PARAM2 = (110, 100, 90, 80, 70, 60, 50, 40, 32)

# A candidate must be reasonably close to the expected anchor.
MAX_RADIUS_ERROR = 0.16
MAX_CENTER_ERROR = 0.28  # relative to radius


def _median(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    if not values:
        return default
    return float(np.median(values))


def annotation_to_box(annotation: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Convert any annotation shape used by the canvas into ``x1,y1,x2,y2``."""
    ann_type = str(annotation.get("type", "rect"))

    if ann_type == "polygon":
        points = annotation.get("points") or []
        xs = [float(p.get("x", 0)) for p in points if isinstance(p, dict)]
        ys = [float(p.get("y", 0)) for p in points if isinstance(p, dict)]
        if len(xs) < 2 or len(ys) < 2:
            return None
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
    else:
        x = float(annotation.get("x", 0))
        y = float(annotation.get("y", 0))
        width = float(annotation.get("width", annotation.get("w", 0)))
        height = float(annotation.get("height", annotation.get("h", 0)))
        x1, y1 = min(x, x + width), min(y, y + height)
        x2, y2 = max(x, x + width), max(y, y + height)

    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def _clip_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, min(float(width), box[0]))
    y1 = max(0.0, min(float(height), box[1]))
    x2 = max(0.0, min(float(width), box[2]))
    y2 = max(0.0, min(float(height), box[3]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def detect_anchor(
    bgr: np.ndarray,
    expected: tuple[float, float, float] | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[float, float, float]:
    """Return ``(cx, cy, r)`` of the anchor.

    ``expected`` is the anchor learned during training (already scaled to the
    image being processed).  When it is omitted a generic guess is used, which
    is what happens while fitting a model from scratch.

    ``config`` comes from the image-training configuration panel.  The default
    is a Hough-circle search tuned for a circular reference object; ``anchor_mode`` can
    be set to ``"fixed"`` for a different camera setup where the region of
    interest does not have a detectable circle.
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("Empty image")

    cfg = normalize_detector_config(config)
    height, width = bgr.shape[:2]
    short = float(min(height, width))
    base = (
        cfg["center_x"] * width,
        cfg["center_y"] * height,
        cfg["radius"] * short,
    )

    if cfg["anchor_mode"] == "fixed":
        if expected is not None:
            return float(expected[0]), float(expected[1]), max(float(expected[2]), 1.0)
        return float(base[0]), float(base[1]), max(float(base[2]), 1.0)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    gray = cv2.medianBlur(gray, 9)

    if expected is None:
        ex, ey, er = base
        min_radius = max(8, int(cfg["radius_min"] * short))
        max_radius = max(min_radius + 2, int(cfg["radius_max"] * short) + 1)
        center_tolerance = cfg["center_tolerance"] * short
    else:
        ex, ey, er = (float(expected[0]), float(expected[1]), float(expected[2]))
        if er <= 1:
            return ex, ey, max(er, 1.0)
        min_radius = max(8, int((1.0 - cfg["radius_error"]) * er))
        max_radius = max(min_radius + 2, int((1.0 + cfg["radius_error"]) * er) + 1)
        center_tolerance = cfg["center_error"] * er

    min_distance = max(12, int(0.40 * short))
    best: tuple[float, int, float, float, float] | None = None

    for param2 in HOUGH_PARAM2:
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=min_distance,
            param1=100,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            continue
        for cx, cy, radius in np.round(circles[0]).astype(float):
            if radius <= 1:
                continue
            radius_error = abs(radius - er) / max(er, 1.0)
            center_error = math.hypot(cx - ex, cy - ey)
            if radius_error > cfg["radius_error"] or center_error > center_tolerance:
                continue
            score = 0.7 * radius_error + 0.3 * (center_error / max(er, 1.0))
            # Prefer the more confident (higher param2) circle on ties.
            score -= param2 * 1e-5
            if best is None or score < best[0]:
                best = (score, param2, cx, cy, radius)

    if best is None:
        return ex, ey, er
    return float(best[2]), float(best[3]), float(best[4])


def anchor_expected(metadata: dict[str, Any], width: int, height: int) -> tuple[float, float, float]:
    anchor = metadata.get("anchor") or {}
    short = float(min(height, width))
    cx = float(anchor.get("cx_norm", DEFAULT_CENTER[0])) * width
    cy = float(anchor.get("cy_norm", DEFAULT_CENTER[1])) * height
    radius = float(anchor.get("r_norm", DEFAULT_RADIUS_RATIO)) * short
    return cx, cy, max(radius, 2.0)


def normalize_box(
    box: tuple[float, float, float, float], anchor: tuple[float, float, float]
) -> tuple[float, float, float, float]:
    """Return ``dx, dy, w, h`` relative to the anchor centre/radius."""
    ax, ay, ar = anchor
    x1, y1, x2, y2 = box
    return (
        ((x1 + x2) * 0.5 - ax) / ar,
        ((y1 + y2) * 0.5 - ay) / ar,
        (x2 - x1) / ar,
        (y2 - y1) / ar,
    )


def box_from_normalized(
    normalized: tuple[float, float, float, float],
    anchor: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    dx, dy, width, height = normalized
    ax, ay, ar = anchor
    cx = ax + dx * ar
    cy = ay + dy * ar
    half_w = width * ar * 0.5
    half_h = height * ar * 0.5
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def canonical_order(
    boxes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Order normalized boxes top-to-bottom, left-to-right within a row.

    This keeps annotations consistent even if the user drew the boxes in a
    different order on different images.
    """
    items = sorted(boxes, key=lambda b: (b[1], b[0]))
    rows: list[list[tuple[float, float, float, float]]] = []
    for item in items:
        if not rows or abs(item[1] - rows[-1][-1][1]) > 0.18:
            rows.append([item])
        else:
            rows[-1].append(item)

    ordered: list[tuple[float, float, float, float]] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b[0]))
    return ordered


def make_layout(samples: list[list[tuple[float, float, float, float]]]) -> list[dict[str, float]]:
    """Median normalized layout from per-image normalized boxes."""
    if not samples:
        raise ValueError("No training samples")
    count = len(samples[0])
    layout: list[dict[str, float]] = []
    for index in range(count):
        values = np.array([sample[index] for sample in samples], dtype=np.float64)
        dx, dy, width, height = np.median(values, axis=0)
        layout.append({
            "dx": float(dx),
            "dy": float(dy),
            "w": float(width),
            "h": float(height),
        })
    return layout


def predict_boxes_with_anchor(
    layout: list[dict[str, float]],
    anchor: tuple[float, float, float],
    width: int,
    height: int,
) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for slot in layout:
        raw = box_from_normalized(
            (slot["dx"], slot["dy"], slot["w"], slot["h"]), anchor
        )
        clipped = _clip_box(raw, width, height)
        # Never drop a slot: the detector must always return K boxes.
        boxes.append(clipped if clipped is not None else raw)
    return boxes


# Circle search used to keep a learned layout on the objects when the camera
# moves or when the plate is rotated.  The radius window rejects the small
# holes inside the wells and the large fixture around them.
_CIRCLE_PARAM2 = (70, 55, 42)
_CIRCLE_MIN_RATIO = 0.10
_CIRCLE_MAX_RATIO = 0.24


def _rect_center(
    box: tuple[float, float, float, float],
) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _rect_radius(box: tuple[float, float, float, float]) -> float:
    return max(1.0, ((box[2] - box[0]) + (box[3] - box[1])) / 4.0)


def _rotate_box_180(
    box: tuple[float, float, float, float], cx: float, cy: float
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return 2 * cx - x2, 2 * cy - y2, 2 * cx - x1, 2 * cy - y1


def _circle_candidates(
    bgr: np.ndarray, min_candidates: int = 6
) -> tuple[list[tuple[float, float, float]], float]:
    height, width = bgr.shape[:2]
    short = float(min(height, width))
    if short <= 0:
        return [], short

    scale = min(1.0, 600.0 / short)
    if scale < 1.0:
        work = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = bgr
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
    gray = cv2.createCLAHE(clipLimit=2, tileGridSize=(2, 2)).apply(gray)
    gray = cv2.medianBlur(gray, 5)

    work_short = float(min(work.shape[:2]))
    min_radius = max(6, int(work_short / 50))
    max_radius = max(min_radius + 2, int(work_short / 2.5))
    min_distance = max(8, int(work_short / 9))

    raw: list[tuple[float, float, float]] = []
    for param2 in _CIRCLE_PARAM2:
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_distance,
            param1=100,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is not None:
            for x, y, radius in np.round(circles[0]).astype(float):
                raw.append((x / scale, y / scale, radius / scale))
        plausible = sum(
            1
            for _x, _y, radius in raw
            if _CIRCLE_MIN_RATIO * short <= radius <= _CIRCLE_MAX_RATIO * short
        )
        if plausible >= min_candidates:
            break

    picked: list[tuple[float, float, float]] = []
    for x, y, radius in raw:
        if not (_CIRCLE_MIN_RATIO * short <= radius <= _CIRCLE_MAX_RATIO * short):
            continue
        duplicate = False
        for index, (px, py, pr) in enumerate(picked):
            if math.hypot(x - px, y - py) < 0.05 * short and 0.6 < radius / pr < 1.7:
                duplicate = True
                if radius > pr:
                    picked[index] = (x, y, radius)
                break
        if not duplicate:
            picked.append((x, y, radius))
    return picked, short


def _match_centers(
    candidates: list[tuple[float, float, float]],
    centers: list[tuple[float, float]],
    expected_radii: list[float],
    short: float,
) -> tuple[dict[int, tuple[float, float, float]], float]:
    sigma = max(2.0, 0.09 * short)
    max_distance = 0.13 * short
    pairs = []
    for center_index, (cx, cy) in enumerate(centers):
        expected_radius = expected_radii[center_index]
        for candidate_index, (x, y, radius) in enumerate(candidates):
            distance = math.hypot(x - cx, y - cy)
            if distance > max_distance:
                continue
            ratio = radius / expected_radius if expected_radius > 0 else 1.0
            if ratio < 0.65 or ratio > 1.8:
                continue
            cost = (distance / sigma) ** 2
            cost += 0.5 * math.log(max(ratio, 1e-6)) ** 2
            pairs.append((cost, distance, center_index, candidate_index, (x, y, radius)))

    pairs.sort(key=lambda item: item[0])
    used_centers: set[int] = set()
    used_candidates: set[int] = set()
    matches: dict[int, tuple[float, float, float]] = {}
    score = 0.0
    for _cost, distance, center_index, candidate_index, circle in pairs:
        if center_index in used_centers or candidate_index in used_candidates:
            continue
        used_centers.add(center_index)
        used_candidates.add(candidate_index)
        matches[center_index] = circle
        quotient = distance / sigma
        if quotient < 1.0:
            score += 1.0 - quotient * quotient
    return matches, score


def _refine_boxes_with_circles(
    metadata: dict[str, Any],
    bgr: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    anchor: tuple[float, float, float],
) -> list[tuple[float, float, float, float]]:
    """Snap predicted boxes onto visible circles and try both orientations.

    The learned layout is a rigid template.  A small camera shift or a
    half-turn of the plate would otherwise leave the template next to the
    wells.  The template is kept as the fallback, so non-circular objects and
    frames without a clear circle behave exactly as before.
    """
    config = metadata.get("detector_config") or {}
    if config.get("refine_with_circles", True) is False:
        return boxes

    height, width = bgr.shape[:2]
    short = float(min(height, width))
    if short <= 0:
        return boxes

    candidates, _short = _circle_candidates(bgr)
    if not candidates:
        return boxes

    # Large overview boxes (the "main" box in some projects) are not wells.
    matchable = [
        index for index, box in enumerate(boxes)
        if _rect_radius(box) <= _CIRCLE_MAX_RATIO * short
    ]
    if not matchable:
        return boxes

    cx, cy = anchor[0], anchor[1]
    variants = [
        list(boxes),
        [_rotate_box_180(box, cx, cy) for box in boxes],
    ]

    best = None
    for variant_index, variant in enumerate(variants):
        centers = [_rect_center(variant[index]) for index in matchable]
        radii = [_rect_radius(variant[index]) for index in matchable]
        matches, score = _match_centers(candidates, centers, radii, short)
        score += 0.001 * len(matches)
        # Prefer the stored orientation when both orientations fit equally.
        if variant_index == 0:
            score += 1e-6
        if best is None or score > best[0]:
            best = (score, variant, matches)

    _score, variant, matches = best
    if not matches:
        return boxes

    refined = list(variant)
    for match_index, circle in matches.items():
        box_index = matchable[match_index]
        rect = variant[box_index]
        x, y, radius = circle
        box_w = max(2.0, rect[2] - rect[0])
        box_h = max(2.0, rect[3] - rect[1])
        box_w = max(box_w, min(2.0 * radius * 1.04, box_w * 1.6))
        box_h = max(box_h, min(2.0 * radius * 1.04, box_h * 1.6))
        snapped = _clip_box(
            (x - box_w / 2, y - box_h / 2, x + box_w / 2, y + box_h / 2),
            width,
            height,
        )
        refined[box_index] = snapped if snapped is not None else rect
    return refined


def predict_boxes(
    metadata: dict[str, Any], bgr: np.ndarray
) -> tuple[list[dict[str, float]], tuple[float, float, float]]:
    """Return exactly ``count`` boxes for the image plus the used anchor."""
    layout = metadata.get("layout") or []
    if not layout:
        raise ValueError("Model has no learned layout")

    height, width = bgr.shape[:2]
    anchor = detect_anchor(
        bgr,
        anchor_expected(metadata, width, height),
        metadata.get("detector_config"),
    )
    boxes = predict_boxes_with_anchor(layout, anchor, width, height)
    boxes = _refine_boxes_with_circles(metadata, bgr, boxes, anchor)
    expected_count = int(metadata.get("count", len(layout)))
    if len(boxes) != expected_count:
        # Clipping should not remove a slot, but never return the wrong count.
        fallback_anchor = anchor_expected(metadata, width, height)
        boxes = predict_boxes_with_anchor(layout, fallback_anchor, width, height)

    result = [
        {
            "x": float(box[0]),
            "y": float(box[1]),
            "w": float(box[2] - box[0]),
            "h": float(box[3] - box[1]),
        }
        for box in boxes
    ]
    return result, anchor


def box_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_and_score(
    predictions: list[tuple[float, float, float, float]],
    ground_truth: list[tuple[float, float, float, float]],
) -> tuple[float, int, list[float]]:
    """Greedy one-to-one IoU matching.  Returns mean IoU and match count."""
    used = [False] * len(predictions)
    ious: list[float] = []
    for gt in ground_truth:
        best_index = -1
        best_iou = 0.0
        for index, pred in enumerate(predictions):
            if used[index]:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0:
            used[best_index] = True
            ious.append(best_iou)
        else:
            ious.append(0.0)
    mean_iou = float(sum(ious) / len(ious)) if ious else 0.0
    return mean_iou, sum(1 for v in ious if v >= 0.5), ious


def predict_and_score(
    metadata: dict[str, Any],
    bgr: np.ndarray,
    ground_truth: list[tuple[float, float, float, float]],
) -> tuple[float, list[dict[str, float]]]:
    detections, _ = predict_boxes(metadata, bgr)
    predicted = [
        (d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]) for d in detections
    ]
    mean_iou, _, _ = match_and_score(predicted, ground_truth)
    return mean_iou, detections


def read_image(path: str) -> np.ndarray:
    """Read an image from a path that may contain non-ASCII characters."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        image = None
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image
