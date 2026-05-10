from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    """Detection result for one hydraulic block candidate."""

    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    area: float
    depth_m: float


class HydraulicBlockDetector:
    """Rule-based detector using depth threshold + largest contour.

    Assumption for the first home experiment:
    - Put the hydraulic block on a simple table/background.
    - Place the camera so the block is the closest large object in view.
    - Tune min_depth_m/max_depth_m if the table or your hand is detected.
    """

    def __init__(
        self,
        min_depth_m=0.15,
        max_depth_m=1.20,
        min_area_px=1200,
        blur_kernel=5,
    ):
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.min_area_px = min_area_px
        self.blur_kernel = blur_kernel

    def detect(self, color_image, depth_image, depth_scale, depth_frame):
        """Detect the biggest object inside the selected depth range."""
        if color_image is None or depth_image is None:
            return None, None

        depth_m = depth_image.astype(np.float32) * depth_scale

        # RealSense returns 0 where depth is invalid. Remove those pixels first.
        valid_depth = depth_m > 0
        in_range = (
            valid_depth
            & (depth_m >= self.min_depth_m)
            & (depth_m <= self.max_depth_m)
        )

        mask = (in_range.astype(np.uint8)) * 255

        # Clean noisy depth pixels before contour detection.
        if self.blur_kernel > 1:
            mask = cv2.medianBlur(mask, self.blur_kernel)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None, mask

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.min_area_px:
            return None, mask

        x, y, w, h = cv2.boundingRect(largest)

        moments = cv2.moments(largest)
        if moments["m00"] != 0:
            center_u = int(moments["m10"] / moments["m00"])
            center_v = int(moments["m01"] / moments["m00"])
        else:
            center_u = x + w // 2
            center_v = y + h // 2

        # get_distance expects pixel coordinate order: (u, v) = (x, y).
        center_depth_m = float(depth_frame.get_distance(center_u, center_v))

        detection = Detection(
            bbox=(x, y, w, h),
            center=(center_u, center_v),
            area=area,
            depth_m=center_depth_m,
        )
        return detection, mask


def draw_detection(color_image, detection):
    """Draw bounding box, center dot, and text on a color image."""
    output = color_image.copy()

    if detection is None:
        cv2.putText(
            output,
            "No block detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    x, y, w, h = detection.bbox
    u, v = detection.center

    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(output, (u, v), 5, (0, 0, 255), -1)

    label = f"center=({u}, {v}), depth={detection.depth_m:.2f}m"
    cv2.putText(
        output,
        label,
        (x, max(30, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output
