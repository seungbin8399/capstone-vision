import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from realsense_block_detector.slot_grid import (
    BlockDetection,
    assign_slot_statuses,
    bbox_center,
    build_result,
    build_slots,
    clamp_bbox,
    detect_box_bbox,
    detect_image_block_candidates,
    draw_debug_image,
    parse_bbox,
    save_result_json,
    xywh_to_xyxy,
    yolo_results_to_block_detections,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect box slots, assign hydraulic block occupancy, and save JSON/debug output."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Path to a still image for offline testing.")
    input_group.add_argument(
        "--camera",
        action="store_true",
        help="Use Intel RealSense D435 color/depth stream.",
    )

    parser.add_argument("--rows", type=int, default=2, help="Grid row count.")
    parser.add_argument("--cols", type=int, default=2, help="Grid column count.")
    parser.add_argument(
        "--box-bbox",
        help="Manual outer box bbox as x1,y1,x2,y2. If omitted, contour detection is used.",
    )
    parser.add_argument(
        "--block-bbox",
        action="append",
        default=[],
        help="Manual hydraulic block bbox as x1,y1,x2,y2. Can be used multiple times.",
    )
    parser.add_argument(
        "--no-image-block-auto",
        action="store_true",
        help="Disable still-image contour fallback for hydraulic block candidates.",
    )
    parser.add_argument(
        "--yolo-model",
        help="Optional Ultralytics YOLO model path, e.g. best.pt from the reference repo workflow.",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.35)

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--min-depth", type=float, default=0.15)
    parser.add_argument("--max-depth", type=float, default=1.20)
    parser.add_argument("--min-area", type=float, default=1200)

    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--output-json", help="Override JSON output path.")
    parser.add_argument("--debug-image", help="Override debug image output path.")
    parser.add_argument(
        "--save-interval",
        type=float,
        default=1.0,
        help="Camera mode save interval in seconds.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open OpenCV preview windows in camera mode.",
    )
    return parser.parse_args()


def make_output_paths(args, stem: str) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    json_path = Path(args.output_json) if args.output_json else output_dir / f"{stem}_grid_result.json"
    debug_path = Path(args.debug_image) if args.debug_image else output_dir / f"{stem}_grid_debug.jpg"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    return json_path, debug_path


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image) -> None:
    ext = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Could not encode debug image as {ext}: {path}")
    encoded.tofile(str(path))


def load_yolo_model(model_path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for --yolo-model. Install it with: pip install ultralytics"
        ) from exc
    return YOLO(model_path)


def detect_yolo_blocks(
    image,
    model,
    conf_threshold: float,
    depth_frame=None,
    intrinsics=None,
    rs_module=None,
) -> list[BlockDetection]:
    results = model.predict(image, conf=conf_threshold, verbose=False)
    return yolo_results_to_block_detections(
        results=results,
        model_names=getattr(model, "names", {}),
        image_shape=image.shape,
        depth_frame=depth_frame,
        intrinsics=intrinsics,
        rs_module=rs_module,
    )


def manual_block_detections(args, image_shape) -> list[BlockDetection]:
    detections = []
    for text in args.block_bbox:
        bbox = clamp_bbox(parse_bbox(text), image_shape)
        detections.append(
            BlockDetection(
                bbox=bbox,
                center=bbox_center(bbox),
                label="manual",
            )
        )
    return detections


def build_slot_result(image, args, block_detections: list[BlockDetection]):
    if args.box_bbox:
        box_bbox = clamp_bbox(parse_bbox(args.box_bbox), image.shape)
    else:
        box_bbox = detect_box_bbox(image)

    slots = build_slots(box_bbox, rows=args.rows, cols=args.cols)
    slots, target_slot = assign_slot_statuses(slots, block_detections)
    result = build_result(
        box_bbox=box_bbox,
        slots=slots,
        target_slot=target_slot,
        block_detections=block_detections,
    )
    debug_image = draw_debug_image(image, box_bbox, slots, block_detections)
    return result, debug_image


def run_image(args):
    image_path = Path(args.image)
    image = read_image(image_path)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    yolo_model = load_yolo_model(args.yolo_model) if args.yolo_model else None
    block_detections = manual_block_detections(args, image.shape)

    if yolo_model is not None:
        block_detections.extend(
            detect_yolo_blocks(image, yolo_model, conf_threshold=args.yolo_conf)
        )

    if not block_detections and not args.no_image_block_auto:
        if args.box_bbox:
            box_bbox = clamp_bbox(parse_bbox(args.box_bbox), image.shape)
        else:
            box_bbox = detect_box_bbox(image)
        block_detections = detect_image_block_candidates(
            image,
            box_bbox,
            min_area_px=args.min_area,
        )

    result, debug_image = build_slot_result(image, args, block_detections)

    json_path, debug_path = make_output_paths(args, image_path.stem)
    save_result_json(result, json_path)
    write_image(debug_path, debug_image)

    print(f"Saved JSON: {json_path}")
    print(f"Saved debug image: {debug_path}")
    print_result_summary(result)


def get_realsense_intrinsics(camera):
    import pyrealsense2 as rs

    profile = camera.pipeline.get_active_profile()
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    return rs, color_profile.get_intrinsics()


def block_detection_from_depth_detection(detection, intrinsics=None, rs_module=None):
    bbox = xywh_to_xyxy(detection.bbox)
    camera_xyz_m = None
    if detection.depth_m > 0 and intrinsics is not None and rs_module is not None:
        xyz = rs_module.rs2_deproject_pixel_to_point(
            intrinsics,
            [int(detection.center[0]), int(detection.center[1])],
            float(detection.depth_m),
        )
        camera_xyz_m = (float(xyz[0]), float(xyz[1]), float(xyz[2]))

    return BlockDetection(
        bbox=bbox,
        center=detection.center,
        label="depth_contour",
        confidence=None,
        depth_m=float(detection.depth_m),
        camera_xyz_m=camera_xyz_m,
    )


def run_camera(args):
    from realsense_block_detector.camera import RealSenseCamera
    from realsense_block_detector.detector import HydraulicBlockDetector

    camera = RealSenseCamera(width=args.width, height=args.height, fps=args.fps)
    detector = HydraulicBlockDetector(
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        min_area_px=args.min_area,
    )
    yolo_model = load_yolo_model(args.yolo_model) if args.yolo_model else None

    json_path, debug_path = make_output_paths(args, "camera")
    last_save_time = 0.0

    try:
        depth_scale = camera.start()
        rs_module, intrinsics = get_realsense_intrinsics(camera)
        print(f"RealSense started. depth_scale={depth_scale}")
        print("Press 'q' or ESC in the OpenCV window to quit.")

        while True:
            color_image, depth_image, depth_frame = camera.get_frames()
            if color_image is None:
                continue

            block_detections = manual_block_detections(args, color_image.shape)
            if yolo_model is not None:
                block_detections.extend(
                    detect_yolo_blocks(
                        color_image,
                        yolo_model,
                        conf_threshold=args.yolo_conf,
                        depth_frame=depth_frame,
                        intrinsics=intrinsics,
                        rs_module=rs_module,
                    )
                )
            elif not block_detections:
                detection, _ = detector.detect(
                    color_image=color_image,
                    depth_image=depth_image,
                    depth_scale=depth_scale,
                    depth_frame=depth_frame,
                )
                if detection is not None:
                    block_detections.append(
                        block_detection_from_depth_detection(
                            detection,
                            intrinsics=intrinsics,
                            rs_module=rs_module,
                        )
                    )

            result, debug_image = build_slot_result(color_image, args, block_detections)

            now = time.time()
            if now - last_save_time >= args.save_interval:
                save_result_json(result, json_path)
                write_image(debug_path, debug_image)
                print_result_summary(result)
                print(f"Saved JSON: {json_path}")
                print(f"Saved debug image: {debug_path}")
                last_save_time = now

            if not args.no_show:
                cv2.imshow("Grid Slot Occupancy", debug_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
            else:
                time.sleep(0.01)

    except RuntimeError as exc:
        print("RealSense error:")
        print(exc)
        print()
        print("Check that the D435 is connected, RealSense Viewer is closed, and streams work.")
    finally:
        camera.stop()
        if not args.no_show:
            cv2.destroyAllWindows()


def print_result_summary(result: dict) -> None:
    slot_text = ", ".join(
        f"{slot['slot_id']}={slot['status']}" for slot in result["slots"]
    )
    target = result["target_slot"]
    target_text = "none" if target is None else f"{target['slot_id']}@{target['center']}"
    print(f"slots: {slot_text} | target: {target_text}")


def main():
    args = parse_args()
    if args.image:
        run_image(args)
    else:
        run_camera(args)


if __name__ == "__main__":
    main()
