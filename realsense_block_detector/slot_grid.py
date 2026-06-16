from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class BlockDetection:
    """Hydraulic block detection normalized for slot assignment."""

    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    label: str = "block"
    confidence: Optional[float] = None
    depth_m: Optional[float] = None
    camera_xyz_m: Optional[tuple[float, float, float]] = None
    polygon: Optional[tuple[tuple[int, int], ...]] = None


@dataclass(frozen=True)
class Slot:
    slot_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    status: str = "empty"
    polygon: Optional[tuple[tuple[int, int], ...]] = None


def clamp_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    return x1, y1, x2, y2


def parse_bbox(text: str) -> tuple[int, int, int, int]:
    parts = [int(float(part.strip())) for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have four comma-separated values: x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must satisfy x2 > x1 and y2 > y1")
    return x1, y1, x2, y2


def parse_quad(text: str) -> tuple[tuple[int, int], ...]:
    parts = [int(float(part.strip())) for part in text.split(",")]
    if len(parts) != 8:
        raise ValueError(
            "quad must have eight comma-separated values: "
            "tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y"
        )
    points = tuple((parts[i], parts[i + 1]) for i in range(0, 8, 2))
    area = cv2.contourArea(np.array(points, dtype=np.float32))
    if area <= 1.0:
        raise ValueError("quad area is too small; click four corners around the box")
    return points


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2


def polygon_center(points: Iterable[tuple[int, int]]) -> tuple[int, int]:
    array = np.array(list(points), dtype=np.float32)
    center = array.mean(axis=0)
    return int(round(float(center[0]))), int(round(float(center[1])))


def quad_to_bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    array = np.array(list(points), dtype=np.int32)
    x, y, w, h = cv2.boundingRect(array)
    return x, y, x + w, y + h


def clamp_quad(
    points: Iterable[tuple[int, int]],
    image_shape: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    height, width = image_shape[:2]
    clamped = []
    for x, y in points:
        clamped.append(
            (
                max(0, min(int(round(x)), width - 1)),
                max(0, min(int(round(y)), height - 1)),
            )
        )
    return tuple(clamped)


def xywh_to_xyxy(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def point_in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x < x2 and y1 <= y < y2


def point_in_polygon(point: tuple[int, int], polygon: Iterable[tuple[int, int]]) -> bool:
    contour = np.array(list(polygon), dtype=np.int32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def point_in_slot(point: tuple[int, int], slot: Slot) -> bool:
    if slot.polygon is not None:
        return point_in_polygon(point, slot.polygon)
    return point_in_bbox(point, slot.bbox)


def slot_detection_overlap_ratio(slot: Slot, detection: BlockDetection) -> float:
    slot_polygon = slot.polygon
    if slot_polygon is None:
        x1, y1, x2, y2 = slot.bbox
        slot_polygon = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))

    slot_contour = np.array(slot_polygon, dtype=np.float32)
    detection_polygon = detection.polygon
    if detection_polygon is None:
        dx1, dy1, dx2, dy2 = detection.bbox
        detection_polygon = ((dx1, dy1), (dx2, dy1), (dx2, dy2), (dx1, dy2))
    detection_contour = np.array(detection_polygon, dtype=np.float32)

    slot_area = cv2.contourArea(slot_contour)
    if slot_area <= 0:
        return 0.0

    overlap_area, _ = cv2.intersectConvexConvex(slot_contour, detection_contour)
    return float(overlap_area) / float(slot_area)


def detect_box_bbox(
    image: np.ndarray,
    min_area_ratio: float = 0.08,
) -> tuple[int, int, int, int]:
    """Detect the outer box/tray as the largest rectangular contour.

    If the image is too clean or contour detection fails, the full image is used
    as a conservative fallback so the slot pipeline can still produce output.
    """
    height, width = image.shape[:2]
    image_area = height * width

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * min_area_ratio:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 20 or h < 20:
            continue

        rect_area = w * h
        fill_ratio = area / float(rect_area)
        score = area * min(fill_ratio, 1.0)
        candidates.append((score, (x, y, x + w, y + h)))

    if candidates:
        _, bbox = max(candidates, key=lambda item: item[0])
        return clamp_bbox(bbox, image.shape)

    return 0, 0, width, height


def build_slots(
    box_bbox: tuple[int, int, int, int],
    rows: int,
    cols: int,
) -> list[Slot]:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive integers")

    x1, y1, x2, y2 = box_bbox
    box_w = x2 - x1
    box_h = y2 - y1

    slots = []
    slot_id = 1
    for row in range(rows):
        sy1 = y1 + round(box_h * row / rows)
        sy2 = y1 + round(box_h * (row + 1) / rows)
        for col in range(cols):
            sx1 = x1 + round(box_w * col / cols)
            sx2 = x1 + round(box_w * (col + 1) / cols)
            bbox = (sx1, sy1, sx2, sy2)
            slots.append(Slot(slot_id=slot_id, bbox=bbox, center=bbox_center(bbox)))
            slot_id += 1
    return slots


def build_quad_slots(
    box_quad: Iterable[tuple[int, int]],
    rows: int,
    cols: int,
) -> list[Slot]:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive integers")

    tl, tr, br, bl = np.array(list(box_quad), dtype=np.float32)

    def interpolate(row_fraction: float, col_fraction: float) -> np.ndarray:
        left = tl + (bl - tl) * row_fraction
        right = tr + (br - tr) * row_fraction
        return left + (right - left) * col_fraction

    slots = []
    slot_id = 1
    for row in range(rows):
        row_top = row / rows
        row_bottom = (row + 1) / rows
        for col in range(cols):
            col_left = col / cols
            col_right = (col + 1) / cols
            polygon_float = np.array(
                [
                    interpolate(row_top, col_left),
                    interpolate(row_top, col_right),
                    interpolate(row_bottom, col_right),
                    interpolate(row_bottom, col_left),
                ],
                dtype=np.float32,
            )
            polygon = tuple(
                (int(round(float(point[0]))), int(round(float(point[1]))))
                for point in polygon_float
            )
            bbox = quad_to_bbox(polygon)
            slots.append(
                Slot(
                    slot_id=slot_id,
                    bbox=bbox,
                    center=polygon_center(polygon),
                    polygon=polygon,
                )
            )
            slot_id += 1
    return slots


def assign_slot_statuses(
    slots: Iterable[Slot],
    block_detections: Iterable[BlockDetection],
    occupancy_mode: str = "center",
    min_slot_overlap: float = 0.20,
) -> tuple[list[Slot], Optional[Slot]]:
    detections = list(block_detections)
    updated = []

    for slot in slots:
        if occupancy_mode == "center":
            occupied = any(point_in_slot(detection.center, slot) for detection in detections)
        elif occupancy_mode == "bbox-overlap":
            occupied = any(
                slot_detection_overlap_ratio(slot, detection) >= min_slot_overlap
                for detection in detections
            )
        else:
            raise ValueError("occupancy_mode must be one of: center, bbox-overlap")
        updated.append(
            Slot(
                slot_id=slot.slot_id,
                bbox=slot.bbox,
                center=slot.center,
                status="occupied" if occupied else "empty",
                polygon=slot.polygon,
            )
        )

    target_slot = next((slot for slot in updated if slot.status == "empty"), None)
    if target_slot is None:
        return updated, None

    updated = [
        Slot(
            slot_id=slot.slot_id,
            bbox=slot.bbox,
            center=slot.center,
            status="target" if slot.slot_id == target_slot.slot_id else slot.status,
            polygon=slot.polygon,
        )
        for slot in updated
    ]
    target_slot = next(slot for slot in updated if slot.status == "target")
    return updated, target_slot


def build_result(
    box_bbox: tuple[int, int, int, int],
    slots: list[Slot],
    target_slot: Optional[Slot],
    block_detections: Optional[list[BlockDetection]] = None,
    box_quad: Optional[tuple[tuple[int, int], ...]] = None,
) -> dict:
    block_detections = block_detections or []
    result = {
        "box": {
            "bbox": list(box_bbox),
            "center": list(bbox_center(box_bbox)),
            "quad": None if box_quad is None else [list(point) for point in box_quad],
        },
        "slots": [
            {
                "slot_id": slot.slot_id,
                "bbox": list(slot.bbox),
                "center": list(slot.center),
                "status": slot.status,
                "polygon": None
                if slot.polygon is None
                else [list(point) for point in slot.polygon],
            }
            for slot in slots
        ],
        "target_slot": None
        if target_slot is None
        else {
            "slot_id": target_slot.slot_id,
            "center": list(target_slot.center),
        },
        "blocks": [
            {
                "bbox": list(detection.bbox),
                "center": list(detection.center),
                "label": detection.label,
                "confidence": detection.confidence,
                "depth_m": detection.depth_m,
                "camera_xyz_m": None
                if detection.camera_xyz_m is None
                else list(detection.camera_xyz_m),
                "polygon": None
                if detection.polygon is None
                else [list(point) for point in detection.polygon],
            }
            for detection in block_detections
        ],
        "robot_target_placeholder": {
            "frame_id": "camera_link",
            "target_slot_center_pixel": None
            if target_slot is None
            else list(target_slot.center),
            "target_slot_camera_xyz_m": None,
            "target_slot_base_xyz_m": None,
            "ros2_topics_todo": [
                "/target_slot_pixel",
                "/target_slot_pose",
            ],
            "note": "Vision-only output. MoveIt/robot execution is intentionally not performed here.",
        },
    }
    return result


def save_result_json(result: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def detect_image_block_candidates(
    image: np.ndarray,
    box_bbox: tuple[int, int, int, int],
    min_area_px: float = 800.0,
) -> list[BlockDetection]:
    """Heuristic fallback for still images when no YOLO/model result is provided."""
    x1, y1, x2, y2 = box_bbox
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 160)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    box_area = (x2 - x1) * (y2 - y1)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_px or area > box_area * 0.70:
            continue

        rx, ry, rw, rh = cv2.boundingRect(contour)
        if rw < 15 or rh < 15:
            continue

        bbox = (x1 + rx, y1 + ry, x1 + rx + rw, y1 + ry + rh)
        detections.append(
            BlockDetection(
                bbox=bbox,
                center=bbox_center(bbox),
                label="image_contour",
                confidence=None,
            )
        )

    return detections


def yolo_results_to_block_detections(
    results,
    model_names,
    image_shape: tuple[int, ...],
    depth_frame=None,
    intrinsics=None,
    rs_module=None,
) -> list[BlockDetection]:
    """Convert Ultralytics YOLO results to normalized block detections.

    This is the adapter point for the reference D435/YOLO script:
    YOLO xyxy bbox -> center pixel -> optional depth -> optional camera XYZ.
    """
    if not results:
        return []

    detections = []
    first_result = results[0]
    names = model_names or {}

    for box in first_result.boxes:
        x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
        bbox = clamp_bbox((x1, y1, x2, y2), image_shape)
        center = bbox_center(bbox)
        center_u, center_v = center

        confidence = float(box.conf[0]) if box.conf is not None else None
        class_id = int(box.cls[0]) if box.cls is not None else -1
        if isinstance(names, dict):
            label = str(names.get(class_id, "block"))
        else:
            label = "block"

        depth_m = None
        camera_xyz_m = None
        if depth_frame is not None:
            depth_m = float(depth_frame.get_distance(center_u, center_v))
            if depth_m > 0.0 and intrinsics is not None and rs_module is not None:
                xyz = rs_module.rs2_deproject_pixel_to_point(
                    intrinsics,
                    [center_u, center_v],
                    depth_m,
                )
                camera_xyz_m = (float(xyz[0]), float(xyz[1]), float(xyz[2]))

        detections.append(
            BlockDetection(
                bbox=bbox,
                center=center,
                label=label,
                confidence=confidence,
                depth_m=depth_m,
                camera_xyz_m=camera_xyz_m,
            )
        )

    return detections


def compute_slot_grid_result(
    image: np.ndarray,
    block_detections: list[BlockDetection],
    rows: int = 2,
    cols: int = 2,
    box_bbox: Optional[tuple[int, int, int, int]] = None,
    box_quad: Optional[tuple[tuple[int, int], ...]] = None,
    slot_label_mode: str = "auto",
    occupancy_mode: str = "center",
    min_slot_overlap: float = 0.20,
) -> tuple[dict, np.ndarray]:
    """Run box/grid/slot post-processing on one D435 color frame."""
    if box_quad is not None:
        detected_box_quad = clamp_quad(box_quad, image.shape)
        detected_box_bbox = clamp_bbox(quad_to_bbox(detected_box_quad), image.shape)
    elif box_bbox is None:
        detected_box_quad = None
        detected_box_bbox = detect_box_bbox(image)
    else:
        detected_box_quad = None
        detected_box_bbox = clamp_bbox(box_bbox, image.shape)

    if detected_box_quad is not None:
        slots = build_quad_slots(detected_box_quad, rows=rows, cols=cols)
    else:
        slots = build_slots(detected_box_bbox, rows=rows, cols=cols)
    slots, target_slot = assign_slot_statuses(
        slots,
        block_detections,
        occupancy_mode=occupancy_mode,
        min_slot_overlap=min_slot_overlap,
    )
    result = build_result(
        box_bbox=detected_box_bbox,
        box_quad=detected_box_quad,
        slots=slots,
        target_slot=target_slot,
        block_detections=block_detections,
    )
    debug_image = draw_debug_image(
        image=image,
        box_bbox=detected_box_bbox,
        box_quad=detected_box_quad,
        slots=slots,
        block_detections=block_detections,
        slot_label_mode=slot_label_mode,
    )
    return result, debug_image


def should_draw_slot_label(slot: Slot, slot_count: int, slot_label_mode: str) -> bool:
    if slot_label_mode == "none":
        return False
    if slot_label_mode == "all":
        return True
    if slot_label_mode == "important":
        return slot.status in {"occupied", "target"}
    if slot_label_mode == "auto":
        if slot_count > 36:
            return slot.status in {"occupied", "target"}
        return True
    raise ValueError("slot_label_mode must be one of: auto, all, important, none")


def draw_debug_image(
    image: np.ndarray,
    box_bbox: tuple[int, int, int, int],
    box_quad: Optional[tuple[tuple[int, int], ...]],
    slots: list[Slot],
    block_detections: Iterable[BlockDetection],
    slot_label_mode: str = "auto",
) -> np.ndarray:
    output = image.copy()
    colors = {
        "empty": (170, 170, 170),
        "occupied": (40, 70, 230),
        "target": (40, 180, 70),
    }
    slot_count = len(slots)
    slot_line_thickness = 1 if slot_count > 36 else 2
    slot_font_scale = 0.42 if slot_count > 36 else 0.62

    if box_quad is not None:
        box_contour = np.array(box_quad, dtype=np.int32)
        cv2.polylines(output, [box_contour], isClosed=True, color=(255, 120, 20), thickness=3)
    else:
        x1, y1, x2, y2 = box_bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 120, 20), 3)

    for slot in slots:
        color = colors.get(slot.status, (255, 255, 255))
        if slot.polygon is not None:
            contour = np.array(slot.polygon, dtype=np.int32)
            cv2.polylines(
                output,
                [contour],
                isClosed=True,
                color=color,
                thickness=slot_line_thickness,
            )
            text_x, text_y = slot.center
        else:
            sx1, sy1, sx2, sy2 = slot.bbox
            cv2.rectangle(output, (sx1, sy1), (sx2, sy2), color, slot_line_thickness)
            text_x, text_y = sx1 + 8, sy1 + 28

        if not should_draw_slot_label(slot, slot_count, slot_label_mode):
            continue

        label = f"{slot.slot_id}:{slot.status}"
        text_origin = (text_x + 8, text_y)
        (text_w, text_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            slot_font_scale,
            2,
        )
        cv2.rectangle(
            output,
            (text_origin[0] - 4, text_origin[1] - text_h - 6),
            (text_origin[0] + text_w + 4, text_origin[1] + 5),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            output,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            slot_font_scale,
            color,
            2,
            cv2.LINE_AA,
        )

    for detection in block_detections:
        bx1, by1, bx2, by2 = detection.bbox
        cx, cy = detection.center
        if detection.polygon is not None:
            contour = np.array(detection.polygon, dtype=np.int32)
            cv2.polylines(output, [contour], isClosed=True, color=(0, 220, 255), thickness=2)
        else:
            cv2.rectangle(output, (bx1, by1), (bx2, by2), (0, 220, 255), 2)
        cv2.circle(output, (cx, cy), 5, (255, 0, 255), -1)
        label = detection.label
        if detection.confidence is not None:
            label = f"{label} {detection.confidence:.2f}"
        cv2.putText(
            output,
            label,
            (bx1, max(22, by1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

    return output
