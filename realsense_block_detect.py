import time
import csv
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


# ============================================================
# User-tunable values
# ============================================================

# Choose detection mode:
# - "fixed_roi": use the center of a fixed ROI as the block center candidate.
# - "contour": use color edge/contour detection.
DETECTION_MODE = "fixed_roi"

# Experiment label used in the CSV file name.
# Current default is the stable metal-block experiment without a post-it.
# Examples: "with_postit", "without_postit", "metal_only"
EXPERIMENT_NAME = "without_postit"

# RealSense D435 stream settings.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 30

# Fixed center ROI size.
# These ratio values are used only when USE_MANUAL_ROI = False.
ROI_WIDTH_RATIO = 0.35
ROI_HEIGHT_RATIO = 0.35

# Stable without_postit setup:
# Use manual ROI coordinates instead of ratio/offset ROI.
# The blue ROI should cover the hydraulic block area.
USE_MANUAL_ROI = True

# Manual ROI coordinates: (x1, y1) is top-left, (x2, y2) is bottom-right.
MANUAL_ROI_X1 = 230
MANUAL_ROI_Y1 = 230
MANUAL_ROI_X2 = 420
MANUAL_ROI_Y2 = 440

# Move the ROI center from the image center.
# Positive X moves right, positive Y moves down.
# These values are used only when USE_MANUAL_ROI = False.
ROI_CENTER_OFFSET_X = 0
ROI_CENTER_OFFSET_Y = 70

# Valid RealSense depth range for this experiment.
# Values outside this range are treated as invalid/outlier depth.
MIN_VALID_DEPTH_M = 0.10
MAX_VALID_DEPTH_M = 1.00

# Canny edge thresholds for contour mode.
CANNY_LOW = 50
CANNY_HIGH = 150

# Contour mode filters.
MIN_CONTOUR_AREA = 1200
MIN_BBOX_WIDTH = 30
MIN_BBOX_HEIGHT = 30
MAX_BBOX_WIDTH_RATIO = 0.85
MAX_BBOX_HEIGHT_RATIO = 0.85
MAX_CONTOUR_AREA_RATIO = 0.55
MIN_ASPECT_RATIO = 0.20
MAX_ASPECT_RATIO = 5.00

# If the exact center depth is invalid, use a small median window.
# 2 means 5x5, 4 means 9x9.
DEPTH_MEDIAN_RADIUS = 4

# Fixed ROI mode depth sampling radius.
# 10 means a 21x21 pixel window centered at (sample_u, sample_v).
DEPTH_SAMPLE_RADIUS = 10

# Move only the depth sampling window from the ROI center.
# center_u, center_v: candidate object center for the hydraulic block.
# sample_u, sample_v: actual pixel location used to read depth.
#
# Metallic hydraulic blocks often have unstable depth at the visual center due
# to large holes and reflections. For that reason, the depth sampling point is
# separated from the object-center candidate and can be moved to a more reliable
# metal surface.
#
# Positive X moves right, positive Y moves down.
DEPTH_SAMPLE_OFFSET_X = -20
DEPTH_SAMPLE_OFFSET_Y = -20

# CSV log file. A new file is written each time this script starts.
CSV_LOG_FILE = f"realsense_depth_log_{EXPERIMENT_NAME}.csv"
INTRINSICS_FILE = f"realsense_intrinsics_{EXPERIMENT_NAME}.txt"
LOG_INTERVAL_SEC = 0.5
DEPTH_FILTER_WINDOW = 5


def is_valid_depth(depth_m):
    """Return True when depth is inside the usable experiment range."""
    return MIN_VALID_DEPTH_M <= depth_m <= MAX_VALID_DEPTH_M


def start_realsense_camera():
    """Start RealSense color and depth streams."""
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        rs.format.bgr8,
        FRAME_FPS,
    )
    config.enable_stream(
        rs.stream.depth,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        rs.format.z16,
        FRAME_FPS,
    )

    profile = pipeline.start(config)

    # Align depth coordinates to color image coordinates.
    align = rs.align(rs.stream.color)

    # Depth frames are aligned to the color frame in this script.
    # Therefore these color intrinsics are the values to use later for
    # pixel + depth -> camera-frame 3D point conversion.
    color_stream = profile.get_stream(rs.stream.color)
    color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    return pipeline, align, color_intrinsics


def get_aligned_frames(pipeline, align):
    """Read one aligned color/depth frame pair from RealSense."""
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        return None, None, None

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    return color_image, depth_image, depth_frame


def create_center_roi(image_shape):
    """Return ROI rectangle and ROI center pixel.

    ROI rectangle format:
        (x1, y1, x2, y2)

    Center pixel format:
        (center_u, center_v)
    """
    height, width = image_shape[:2]

    if USE_MANUAL_ROI:
        x1 = max(0, min(MANUAL_ROI_X1, width - 1))
        y1 = max(0, min(MANUAL_ROI_Y1, height - 1))
        x2 = max(1, min(MANUAL_ROI_X2, width))
        y2 = max(1, min(MANUAL_ROI_Y2, height))

        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)

        center_u = (x1 + x2) // 2
        center_v = (y1 + y2) // 2
        return (x1, y1, x2, y2), (center_u, center_v)

    roi_w = int(width * ROI_WIDTH_RATIO)
    roi_h = int(height * ROI_HEIGHT_RATIO)

    center_u = width // 2 + ROI_CENTER_OFFSET_X
    center_v = height // 2 + ROI_CENTER_OFFSET_Y

    # Keep the center inside the image even if offsets are changed too much.
    center_u = max(0, min(center_u, width - 1))
    center_v = max(0, min(center_v, height - 1))

    x1 = center_u - roi_w // 2
    y1 = center_v - roi_h // 2
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    # Shift the rectangle back inside the image while preserving its size.
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > width:
        x1 -= x2 - width
        x2 = width
    if y2 > height:
        y1 -= y2 - height
        y2 = height

    x1 = max(0, x1)
    y1 = max(0, y1)

    return (x1, y1, x2, y2), (center_u, center_v)


def create_center_roi_mask(image_shape):
    """Create a binary mask that keeps only the center ROI."""
    height, width = image_shape[:2]
    roi_rect, _ = create_center_roi(image_shape)
    x1, y1, x2, y2 = roi_rect

    roi_mask = np.zeros((height, width), dtype=np.uint8)
    roi_mask[y1:y2, x1:x2] = 255
    return roi_mask, roi_rect


def make_center_sample_rect(center_u, center_v, radius, image_width, image_height):
    """Create a small depth sampling rectangle around the center pixel."""
    x1 = max(0, center_u - radius)
    y1 = max(0, center_v - radius)
    x2 = min(image_width, center_u + radius + 1)
    y2 = min(image_height, center_v + radius + 1)
    return (x1, y1, x2, y2)


def median_depth_in_rect(depth_frame, sample_rect):
    """Calculate median valid depth inside a sampling rectangle.

    Depth value 0.0 means invalid depth in RealSense, so it is excluded.
    Values outside MIN_VALID_DEPTH_M ~ MAX_VALID_DEPTH_M are also excluded.
    """
    x1, y1, x2, y2 = sample_rect
    values = []

    width = depth_frame.get_width()
    height = depth_frame.get_height()

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    for v in range(y1, y2):
        for u in range(x1, x2):
            depth_m = depth_frame.get_distance(u, v)
            if is_valid_depth(depth_m):
                values.append(depth_m)

    if not values:
        return 0.0

    return float(np.median(values))


def estimate_depth_median(depth_frame, center_u, center_v, radius):
    """Return median valid depth around the center if exact center is invalid."""
    values = []

    width = depth_frame.get_width()
    height = depth_frame.get_height()

    for v in range(center_v - radius, center_v + radius + 1):
        for u in range(center_u - radius, center_u + radius + 1):
            if u < 0 or u >= width or v < 0 or v >= height:
                continue

            depth_m = depth_frame.get_distance(u, v)
            if is_valid_depth(depth_m):
                values.append(depth_m)

    if not values:
        return 0.0

    return float(np.median(values))


def read_depth_at_center(depth_frame, center_u, center_v):
    """Read depth at center pixel, with median fallback for invalid center."""
    depth_m = depth_frame.get_distance(center_u, center_v)

    if not is_valid_depth(depth_m):
        depth_m = estimate_depth_median(
            depth_frame,
            center_u,
            center_v,
            DEPTH_MEDIAN_RADIUS,
        )

    return depth_m


def update_filtered_depth(detection, depth_buffer):
    """Update rolling depth median using the current raw valid depth."""
    if detection is None:
        return None

    raw_depth_m = detection["raw_depth_m"]

    if is_valid_depth(raw_depth_m):
        depth_buffer.append(raw_depth_m)

    if len(depth_buffer) > 0:
        filtered_depth_m = float(np.median(list(depth_buffer)))
    else:
        filtered_depth_m = 0.0

    detection["filtered_depth_m"] = filtered_depth_m
    return detection


def add_camera_xyz(detection, color_intrinsics):
    """Add camera-frame 3D coordinates to a valid detection.

    Formula:
        X = (center_u - cx) * Z / fx
        Y = (center_v - cy) * Z / fy
        Z = filtered_depth_m

    center_u, center_v are the object-center candidate pixels.
    filtered_depth_m is the stabilized depth measured near sample_u, sample_v.
    The result is expressed in the RealSense color camera coordinate frame.
    """
    if detection is None:
        return None

    raw_depth_m = detection["raw_depth_m"]
    filtered_depth_m = detection["filtered_depth_m"]

    if not is_valid_depth(raw_depth_m) or not is_valid_depth(filtered_depth_m):
        detection["camera_xyz_m"] = None
        return detection

    center_u, center_v = detection["center"]
    fx = color_intrinsics.fx
    fy = color_intrinsics.fy
    cx = color_intrinsics.ppx
    cy = color_intrinsics.ppy

    camera_z_m = filtered_depth_m
    camera_x_m = (center_u - cx) * camera_z_m / fx
    camera_y_m = (center_v - cy) * camera_z_m / fy

    detection["camera_xyz_m"] = (camera_x_m, camera_y_m, camera_z_m)
    return detection


def detect_fixed_roi(color_image, depth_frame):
    """Use fixed center ROI as the block center/depth estimate.

    This mode is intentionally simple for early experiments:
    - The blue ROI is placed in the image center.
    - The red point is the ROI center.
    - raw_depth_m is the median of valid depth values near the center point.
    """
    roi_rect, center = create_center_roi(color_image.shape)
    center_u, center_v = center
    image_height, image_width = color_image.shape[:2]

    sample_center_u = center_u + DEPTH_SAMPLE_OFFSET_X
    sample_center_v = center_v + DEPTH_SAMPLE_OFFSET_Y
    sample_center_u = max(0, min(sample_center_u, image_width - 1))
    sample_center_v = max(0, min(sample_center_v, image_height - 1))

    sample_rect = make_center_sample_rect(
        sample_center_u,
        sample_center_v,
        DEPTH_SAMPLE_RADIUS,
        image_width,
        image_height,
    )
    raw_depth_m = median_depth_in_rect(depth_frame, sample_rect)

    return {
        "mode": "fixed_roi",
        "bbox": roi_rect,
        "center": (center_u, center_v),
        "raw_depth_m": raw_depth_m,
        "filtered_depth_m": raw_depth_m,
        "area": (roi_rect[2] - roi_rect[0]) * (roi_rect[3] - roi_rect[1]),
        "roi_rect": roi_rect,
        "sample_rect": sample_rect,
        "sample_center": (sample_center_u, sample_center_v),
    }


def make_color_edge_mask(color_image):
    """Create edge mask for contour mode using grayscale + Canny."""
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    roi_mask, roi_rect = create_center_roi_mask(color_image.shape)
    edges = cv2.bitwise_and(edges, roi_mask)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    return edges, roi_rect


def is_bad_bbox(x, y, w, h, image_width, image_height, area):
    """Reject contour boxes that are too small, too large, or oddly shaped."""
    if w < MIN_BBOX_WIDTH or h < MIN_BBOX_HEIGHT:
        return True

    aspect_ratio = w / float(h)
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return True

    if w >= image_width * MAX_BBOX_WIDTH_RATIO:
        return True
    if h >= image_height * MAX_BBOX_HEIGHT_RATIO:
        return True
    if area >= image_width * image_height * MAX_CONTOUR_AREA_RATIO:
        return True

    return False


def detect_block_from_color(color_mask, depth_frame, roi_rect):
    """Find the best contour from the color edge mask."""
    image_height, image_width = color_mask.shape

    contours, _ = cv2.findContours(
        color_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    image_center_u = image_width / 2.0
    image_center_v = image_height / 2.0
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if is_bad_bbox(x, y, w, h, image_width, image_height, area):
            continue

        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            center_u = int(moments["m10"] / moments["m00"])
            center_v = int(moments["m01"] / moments["m00"])
        else:
            center_u = x + w // 2
            center_v = y + h // 2

        raw_depth_m = read_depth_at_center(depth_frame, center_u, center_v)
        distance_to_center = np.hypot(
            center_u - image_center_u,
            center_v - image_center_v,
        )
        score = area - distance_to_center * 5.0

        candidates.append(
            {
                "mode": "contour",
                "bbox": (x, y, x + w, y + h),
                "center": (center_u, center_v),
                "raw_depth_m": raw_depth_m,
                "filtered_depth_m": raw_depth_m,
                "area": area,
                "aspect_ratio": w / float(h),
                "score": score,
                "roi_rect": roi_rect,
                "sample_rect": None,
                "sample_center": None,
            }
        )

    if not candidates:
        return None

    return max(candidates, key=lambda item: item["score"])


def make_depth_visualization(depth_image):
    """Create a colored depth image for debugging only."""
    depth_8u = cv2.convertScaleAbs(depth_image, alpha=0.03)
    return cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)


def open_csv_logger():
    """Open CSV file and write the header row."""
    log_path = Path(__file__).resolve().parent / CSV_LOG_FILE
    csv_file = log_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "timestamp",
            "center_u",
            "center_v",
            "sample_u",
            "sample_v",
            "raw_depth_m",
            "filtered_depth_m",
            "camera_x_m",
            "camera_y_m",
            "camera_z_m",
            "mode",
        ]
    )
    csv_file.flush()
    return log_path, csv_file, writer


def write_detection_to_csv(writer, csv_file, detection):
    """Save one valid detection row to CSV."""
    center_u, center_v = detection["center"]
    sample_center = detection.get("sample_center")
    if sample_center is None:
        sample_u, sample_v = center_u, center_v
    else:
        sample_u, sample_v = sample_center

    raw_depth_m = detection["raw_depth_m"]
    filtered_depth_m = detection["filtered_depth_m"]
    camera_xyz_m = detection["camera_xyz_m"]
    mode = detection["mode"]
    camera_x_m, camera_y_m, camera_z_m = camera_xyz_m

    writer.writerow(
        [
            f"{time.time():.3f}",
            center_u,
            center_v,
            sample_u,
            sample_v,
            f"{raw_depth_m:.6f}",
            f"{filtered_depth_m:.6f}",
            f"{camera_x_m:.6f}",
            f"{camera_y_m:.6f}",
            f"{camera_z_m:.6f}",
            mode,
        ]
    )
    csv_file.flush()


def save_and_print_intrinsics(color_intrinsics):
    """Print and save color camera intrinsics for the next 3D step.

    The script uses these values for the first camera-frame 3D conversion:

        X = (center_u - cx) * Z / fx
        Y = (center_v - cy) * Z / fy
        Z = filtered_depth_m

    center_u, center_v are the block-center candidate pixels. The depth Z is
    stabilized from the separate depth sampling point.
    """
    intrinsics_path = Path(__file__).resolve().parent / INTRINSICS_FILE

    fx = color_intrinsics.fx
    fy = color_intrinsics.fy
    cx = color_intrinsics.ppx
    cy = color_intrinsics.ppy

    print("Color camera intrinsics for future 3D conversion:")
    print(f"  fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}")
    print("3D formula:")
    print("  X = (center_u - cx) * Z / fx")
    print("  Y = (center_v - cy) * Z / fy")
    print("  Z = filtered_depth_m")
    print(f"  intrinsics file: {intrinsics_path}")

    with intrinsics_path.open("w", encoding="utf-8") as file:
        file.write(f"experiment_name={EXPERIMENT_NAME}\n")
        file.write("stream=color_aligned_depth\n")
        file.write(f"width={color_intrinsics.width}\n")
        file.write(f"height={color_intrinsics.height}\n")
        file.write(f"fx={fx:.9f}\n")
        file.write(f"fy={fy:.9f}\n")
        file.write(f"cx={cx:.9f}\n")
        file.write(f"cy={cy:.9f}\n")
        file.write("formula_x=(center_u-cx)*z/fx\n")
        file.write("formula_y=(center_v-cy)*z/fy\n")
        file.write("formula_z=filtered_depth_m\n")
        file.write(f"model={color_intrinsics.model}\n")
        file.write(
            "coeffs="
            + ",".join(f"{coeff:.9f}" for coeff in color_intrinsics.coeffs)
            + "\n"
        )

    return intrinsics_path


def draw_result(color_image, detection, roi_rect):
    """Draw ROI, center point, depth, mode, and optional contour bbox."""
    result = color_image.copy()

    roi_x1, roi_y1, roi_x2, roi_y2 = roi_rect

    # Blue box: fixed center ROI.
    cv2.rectangle(result, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

    if detection is None:
        cv2.putText(
            result,
            f"mode={DETECTION_MODE}, no detection",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return result

    center_u, center_v = detection["center"]
    raw_depth_m = detection["raw_depth_m"]
    filtered_depth_m = detection["filtered_depth_m"]
    area = detection["area"]
    mode = detection["mode"]

    # Green box is only the contour bbox in contour mode.
    if mode == "contour":
        x1, y1, x2, y2 = detection["bbox"]
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Red dot: selected center pixel.
    cv2.circle(result, (center_u, center_v), 6, (0, 0, 255), -1)

    # Green small box: actual depth sampling window in fixed_roi mode.
    sample_rect = detection.get("sample_rect")
    if sample_rect is not None:
        sx1, sy1, sx2, sy2 = sample_rect
        cv2.rectangle(result, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)

    sample_center = detection.get("sample_center")
    if sample_center is not None:
        sample_u, sample_v = sample_center
        cv2.circle(result, (sample_u, sample_v), 4, (0, 255, 0), -1)

    text1 = f"mode={mode}"
    text2 = f"filtered_depth={filtered_depth_m:.3f}m"
    text3 = f"center=({center_u}, {center_v}), raw_depth={raw_depth_m:.3f}m"
    camera_xyz_m = detection.get("camera_xyz_m")
    if camera_xyz_m is None:
        text4 = "camera_xyz=invalid"
    else:
        camera_x_m, camera_y_m, camera_z_m = camera_xyz_m
        text4 = (
            f"camera_xyz=({camera_x_m:.3f}, "
            f"{camera_y_m:.3f}, {camera_z_m:.3f})m"
        )
    text5 = f"area={area:.0f}"

    cv2.putText(
        result,
        text1,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        text2,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        text3,
        (20, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        text4,
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        text5,
        (20, 152),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return result


def main():
    """Main loop for fixed ROI or contour mode."""
    pipeline = None
    csv_file = None
    log_path = Path(__file__).resolve().parent / CSV_LOG_FILE
    intrinsics_path = Path(__file__).resolve().parent / INTRINSICS_FILE

    try:
        log_path, csv_file, csv_writer = open_csv_logger()

        pipeline, align, color_intrinsics = start_realsense_camera()
        intrinsics_path = save_and_print_intrinsics(color_intrinsics)

        print("RealSense D435 started.")
        print(f"Experiment name: {EXPERIMENT_NAME}")
        print(f"Detection mode: {DETECTION_MODE}")
        print(f"CSV log file: {log_path}")
        print(
            f"Valid depth range: {MIN_VALID_DEPTH_M:.2f}m "
            f"~ {MAX_VALID_DEPTH_M:.2f}m"
        )
        print("Press 'q' or ESC on the OpenCV window to quit.")
        print(f"ROI ratio: width={ROI_WIDTH_RATIO:.2f}, height={ROI_HEIGHT_RATIO:.2f}")

        last_log_time = 0.0
        depth_buffer = deque(maxlen=DEPTH_FILTER_WINDOW)

        while True:
            color_image, depth_image, depth_frame = get_aligned_frames(
                pipeline,
                align,
            )

            if color_image is None:
                continue

            roi_rect, _ = create_center_roi(color_image.shape)

            if DETECTION_MODE == "fixed_roi":
                detection = detect_fixed_roi(color_image, depth_frame)
                color_debug, _ = create_center_roi_mask(color_image.shape)
            elif DETECTION_MODE == "contour":
                color_debug, roi_rect = make_color_edge_mask(color_image)
                detection = detect_block_from_color(color_debug, depth_frame, roi_rect)
            else:
                raise ValueError(
                    'DETECTION_MODE must be "fixed_roi" or "contour".'
                )

            detection = update_filtered_depth(detection, depth_buffer)
            detection = add_camera_xyz(detection, color_intrinsics)

            result_image = draw_result(color_image, detection, roi_rect)
            depth_vis = make_depth_visualization(depth_image)

            cv2.imshow("RealSense Block Detection", result_image)
            cv2.imshow("Color Mask or Edge", color_debug)
            cv2.imshow("Depth Visualization", depth_vis)

            now = time.time()
            if detection is not None and now - last_log_time > LOG_INTERVAL_SEC:
                center_u, center_v = detection["center"]
                raw_depth_m = detection["raw_depth_m"]
                filtered_depth_m = detection["filtered_depth_m"]
                camera_xyz_m = detection["camera_xyz_m"]

                if raw_depth_m <= 0.0:
                    print(
                        f"invalid depth: mode={detection['mode']}, "
                        f"center=({center_u}, {center_v}), depth=0.000m"
                    )
                elif not is_valid_depth(raw_depth_m):
                    print(
                        f"outlier depth: mode={detection['mode']}, "
                        f"center=({center_u}, {center_v}), "
                        f"raw_depth={raw_depth_m:.3f}m"
                    )
                elif camera_xyz_m is None:
                    print(
                        f"invalid camera xyz: mode={detection['mode']}, "
                        f"center=({center_u}, {center_v}), "
                        f"filtered_depth={filtered_depth_m:.3f}m"
                    )
                else:
                    camera_x_m, camera_y_m, camera_z_m = camera_xyz_m
                    write_detection_to_csv(csv_writer, csv_file, detection)
                    print(
                        f"mode={detection['mode']}, "
                        f"center=({center_u}, {center_v}), "
                        f"raw_depth={raw_depth_m:.3f}m, "
                        f"filtered_depth={filtered_depth_m:.3f}m, "
                        f"camera_xyz=({camera_x_m:.3f}, "
                        f"{camera_y_m:.3f}, {camera_z_m:.3f})m, "
                        f"area={detection['area']:.1f}"
                    )

                last_log_time = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    except RuntimeError as error:
        print("RealSense runtime error:")
        print(error)
        print()
        print("Check these points:")
        print("1. D435 is connected to a USB 3.x port.")
        print("2. RealSense Viewer is closed before running this script.")
        print("3. Color Camera and Stereo Module work in RealSense Viewer.")

    finally:
        if pipeline is not None:
            pipeline.stop()
        if csv_file is not None:
            csv_file.close()
        cv2.destroyAllWindows()
        print(f"Saved CSV log file: {log_path}")
        print(f"Saved intrinsics file: {intrinsics_path}")


if __name__ == "__main__":
    main()
