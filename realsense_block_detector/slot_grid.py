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


@dataclass(frozen=True)
class Slot:
    slot_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    status: str = "empty"


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


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2


def xywh_to_xyxy(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def point_in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x < x2 and y1 <= y < y2


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


def assign_slot_statuses(
    slots: Iterable[Slot],
    block_detections: Iterable[BlockDetection],
) -> tuple[list[Slot], Optional[Slot]]:
    detections = list(block_detections)
    updated = []

    for slot in slots:
        occupied = any(point_in_bbox(detection.center, slot.bbox) for detection in detections)
        updated.append(
            Slot(
                slot_id=slot.slot_id,
                bbox=slot.bbox,
                center=slot.center,
                status="occupied" if occupied else "empty",
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
) -> dict:
    block_detections = block_detections or []
    result = {
        "box": {
            "bbox": list(box_bbox),
            "center": list(bbox_center(box_bbox)),
        },
        "slots": [
            {
                "slot_id": slot.slot_id,
                "bbox": list(slot.bbox),
                "center": list(slot.center),
                "status": slot.status,
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
) -> tuple[dict, np.ndarray]:
    """Run box/grid/slot post-processing on one D435 color frame."""
    if box_bbox is None:
        detected_box_bbox = detect_box_bbox(image)
    else:
        detected_box_bbox = clamp_bbox(box_bbox, image.shape)

    slots = build_slots(detected_box_bbox, rows=rows, cols=cols)
    slots, target_slot = assign_slot_statuses(slots, block_detections)
    result = build_result(
        box_bbox=detected_box_bbox,
        slots=slots,
        target_slot=target_slot,
        block_detections=block_detections,
    )
    debug_image = draw_debug_image(
        image=image,
        box_bbox=detected_box_bbox,
        slots=slots,
        block_detections=block_detections,
    )
    return result, debug_image


def draw_debug_image(
    image: np.ndarray,
    box_bbox: tuple[int, int, int, int],
    slots: list[Slot],
    block_detections: Iterable[BlockDetection],
) -> np.ndarray:
    output = image.copy()
    colors = {
        "empty": (170, 170, 170),
        "occupied": (40, 70, 230),
        "target": (40, 180, 70),
    }

    x1, y1, x2, y2 = box_bbox
    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 120, 20), 3)

    for slot in slots:
        sx1, sy1, sx2, sy2 = slot.bbox
        color = colors.get(slot.status, (255, 255, 255))
        cv2.rectangle(output, (sx1, sy1), (sx2, sy2), color, 2)

        label = f"{slot.slot_id}:{slot.status}"
        text_origin = (sx1 + 8, sy1 + 28)
        (text_w, text_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
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
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )

    for detection in block_detections:
        bx1, by1, bx2, by2 = detection.bbox
        cx, cy = detection.center
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
