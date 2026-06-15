import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from realsense_block_detector.slot_grid import (
    BlockDetection,
    bbox_center,
    clamp_bbox,
    compute_slot_grid_result,
    detect_box_bbox,
    detect_image_block_candidates,
    parse_bbox,
    save_result_json,
    yolo_results_to_block_detections,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Lab-ready D435 + YOLO hydraulic block detection with optional "
            "box/grid slot occupancy output."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--camera", action="store_true", help="Use Intel RealSense D435.")
    input_group.add_argument("--image", help="Use a still image for offline testing.")

    parser.add_argument(
        "--yolo-model",
        help="Optional YOLO model path, e.g. best.pt. If omitted, YOLO is skipped.",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument(
        "--enable-slot-grid",
        action="store_true",
        help="Enable box grid/slot occupancy and target slot output.",
    )
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument(
        "--box-bbox",
        help="Manual box bbox as x1,y1,x2,y2. Useful when automatic box detection is unstable.",
    )
    parser.add_argument(
        "--select-box-bbox",
        action="store_true",
        help="Select the box bbox by dragging an ROI on the image or first D435 frame.",
    )
    parser.add_argument(
        "--block-bbox",
        action="append",
        default=[],
        help="Manual block bbox as x1,y1,x2,y2. Can be repeated for fallback tests.",
    )
    parser.add_argument(
        "--no-image-block-auto",
        action="store_true",
        help="Disable image contour fallback for block candidates when YOLO/manual block bbox is absent.",
    )

    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-name", default="lab_slot_grid")
    parser.add_argument("--save-interval", type=float, default=1.0)
    parser.add_argument(
        "--save-latest",
        action="store_true",
        help="Also overwrite latest JSON/debug image names on each save.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open OpenCV preview windows.")
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Show a colorized aligned depth stream in camera mode.",
    )
    parser.add_argument(
        "--ros-publish",
        action="store_true",
        help="Publish slot grid vision results to ROS2 topics.",
    )
    parser.add_argument("--ros-node-name", default="vision_slot_publisher")
    parser.add_argument(
        "--ros-rate",
        type=float,
        default=5.0,
        help="Maximum ROS2 publish rate in Hz.",
    )
    parser.add_argument(
        "--camera-frame",
        default="camera_color_optical_frame",
        help="frame_id for camera-frame PoseStamped/PointStamped messages.",
    )
    return parser.parse_args()


def load_yolo_model(model_path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for --yolo-model. Install with: pip install -r requirements.txt"
        ) from exc
    return YOLO(model_path)


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Could not encode debug image as {ext}: {path}")
    encoded.tofile(str(path))


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


class VisionRosPublisher:
    """Optional ROS2 publisher loaded only when --ros-publish is enabled."""

    def __init__(self, node_name: str, camera_frame: str, publish_rate_hz: float):
        try:
            import rclpy
            from geometry_msgs.msg import PointStamped, PoseStamped
            from std_msgs.msg import String
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 Python packages are required for --ros-publish. "
                "Source your ROS2 setup file and install rclpy/std_msgs/geometry_msgs."
            ) from exc

        self.rclpy = rclpy
        self.PointStamped = PointStamped
        self.PoseStamped = PoseStamped
        self.String = String
        self.camera_frame = camera_frame
        self.min_period_sec = 0.0 if publish_rate_hz <= 0 else 1.0 / publish_rate_hz
        self.last_publish_time = 0.0

        if not self.rclpy.ok():
            self.rclpy.init()

        self.node = self.rclpy.create_node(node_name)
        self.slot_grid_pub = self.node.create_publisher(
            self.String,
            "/vision/slot_grid_result",
            10,
        )
        self.block_pose_pub = self.node.create_publisher(
            self.PoseStamped,
            "/vision/block_pose_camera",
            10,
        )
        self.target_pixel_pub = self.node.create_publisher(
            self.PointStamped,
            "/vision/target_slot_pixel",
            10,
        )
        self.target_pose_pub = self.node.create_publisher(
            self.PoseStamped,
            "/vision/target_slot_pose_camera",
            10,
        )

        domain_id = os.environ.get("ROS_DOMAIN_ID", "not set")
        print(f"ROS2 publish enabled. node={node_name}, frame_id={camera_frame}, ROS_DOMAIN_ID={domain_id}")
        print("Publishing topics:")
        print("  /vision/slot_grid_result       std_msgs/String")
        print("  /vision/block_pose_camera      geometry_msgs/PoseStamped")
        print("  /vision/target_slot_pixel      geometry_msgs/PointStamped")
        print("  /vision/target_slot_pose_camera geometry_msgs/PoseStamped (future target depth)")

    def publish(self, result: dict) -> None:
        now_sec = time.time()
        if now_sec - self.last_publish_time < self.min_period_sec:
            self.rclpy.spin_once(self.node, timeout_sec=0.0)
            return

        stamp = self.node.get_clock().now().to_msg()

        slot_msg = self.String()
        slot_msg.data = json.dumps(result, ensure_ascii=False)
        self.slot_grid_pub.publish(slot_msg)

        block_msg = self._make_first_block_pose(result, stamp)
        if block_msg is not None:
            self.block_pose_pub.publish(block_msg)

        target_pixel_msg = self._make_target_slot_pixel(result, stamp)
        if target_pixel_msg is not None:
            self.target_pixel_pub.publish(target_pixel_msg)

        target_pose_msg = self._make_target_slot_pose(result, stamp)
        if target_pose_msg is not None:
            self.target_pose_pub.publish(target_pose_msg)

        self.last_publish_time = now_sec
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def _make_first_block_pose(self, result: dict, stamp):
        for block in result.get("blocks", []):
            xyz = block.get("camera_xyz_m")
            if xyz is None:
                continue

            msg = self.PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = self.camera_frame
            msg.pose.position.x = float(xyz[0])
            msg.pose.position.y = float(xyz[1])
            msg.pose.position.z = float(xyz[2])
            msg.pose.orientation.x = 0.0
            msg.pose.orientation.y = 0.0
            msg.pose.orientation.z = 0.0
            msg.pose.orientation.w = 1.0
            return msg
        return None

    def _make_target_slot_pixel(self, result: dict, stamp):
        target = result.get("target_slot")
        if target is None:
            return None

        center = target.get("center")
        if center is None:
            return None

        msg = self.PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_frame
        msg.point.x = float(center[0])
        msg.point.y = float(center[1])
        msg.point.z = float(target.get("slot_id", 0))
        return msg

    def _make_target_slot_pose(self, result: dict, stamp):
        placeholder = result.get("robot_target_placeholder", {})
        xyz = placeholder.get("target_slot_camera_xyz_m")
        if xyz is None:
            return None

        msg = self.PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_frame
        msg.pose.position.x = float(xyz[0])
        msg.pose.position.y = float(xyz[1])
        msg.pose.position.z = float(xyz[2])
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        return msg

    def shutdown(self) -> None:
        self.node.destroy_node()
        self.rclpy.shutdown()


def make_output_paths(args, suffix: str = "") -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp()
    suffix_part = f"_{suffix}" if suffix else ""
    json_path = output_dir / f"{args.run_name}_{stamp}{suffix_part}_result.json"
    debug_path = output_dir / f"{args.run_name}_{stamp}{suffix_part}_debug.jpg"
    return json_path, debug_path


def latest_output_paths(args) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        output_dir / f"{args.run_name}_latest_result.json",
        output_dir / f"{args.run_name}_latest_debug.jpg",
    )


def manual_block_detections(args, image_shape) -> list[BlockDetection]:
    detections = []
    for text in args.block_bbox:
        bbox = clamp_bbox(parse_bbox(text), image_shape)
        center = bbox_center(bbox)
        detections.append(BlockDetection(bbox=bbox, center=center, label="manual"))
    return detections


def select_box_bbox_from_image(image, window_name: str) -> str:
    print("Drag the storage box area, then press ENTER or SPACE. Press c to cancel.")
    x, y, w, h = cv2.selectROI(
        window_name,
        image,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow(window_name)

    if w <= 0 or h <= 0:
        raise RuntimeError("Box bbox selection was canceled or empty.")

    bbox = clamp_bbox((x, y, x + w, y + h), image.shape)
    bbox_text = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    print(f"Selected box bbox: --box-bbox {bbox_text}")
    return bbox_text


def detect_yolo_blocks(image, model, args, depth_frame=None, intrinsics=None, rs_module=None):
    if model is None:
        return [], image.copy()

    results = model.predict(image, conf=args.conf, verbose=False)
    annotated = results[0].plot() if results else image.copy()
    detections = yolo_results_to_block_detections(
        results=results,
        model_names=model.names,
        image_shape=image.shape,
        depth_frame=depth_frame,
        intrinsics=intrinsics,
        rs_module=rs_module,
    )
    return detections, annotated


def print_block_summary(block_detections: list[BlockDetection]) -> None:
    if not block_detections:
        print("blocks: none")
        return

    for detection in block_detections:
        cx, cy = detection.center
        conf_text = "n/a" if detection.confidence is None else f"{detection.confidence:.2f}"
        depth_text = "n/a" if detection.depth_m is None else f"{detection.depth_m:.3f}m"
        xyz = detection.camera_xyz_m
        xyz_text = "n/a" if xyz is None else f"({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})m"
        print(
            f"block label={detection.label} conf={conf_text} "
            f"center=({cx},{cy}) depth={depth_text} xyz={xyz_text}"
        )


def print_slot_summary(result: dict) -> None:
    target = result["target_slot"]
    target_text = "none" if target is None else f"{target['slot_id']} center={target['center']}"
    statuses = ", ".join(
        f"{slot['slot_id']}:{slot['status']}" for slot in result["slots"]
    )
    print(f"slot_status=[{statuses}] target={target_text}")


def make_depth_visualization(depth_frame) -> np.ndarray:
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_8u = cv2.convertScaleAbs(depth_image, alpha=0.03)
    return cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)


def save_outputs(args, result: dict, debug_image, suffix: str = "") -> tuple[Path, Path]:
    json_path, debug_path = make_output_paths(args, suffix=suffix)
    save_result_json(result, json_path)
    write_image(debug_path, debug_image)

    if args.save_latest:
        latest_json, latest_debug = latest_output_paths(args)
        save_result_json(result, latest_json)
        write_image(latest_debug, debug_image)

    return json_path, debug_path


def maybe_publish_ros(ros_publisher, result: dict | None) -> None:
    if ros_publisher is not None and result is not None:
        ros_publisher.publish(result)


def add_image_fallback_blocks(args, image, block_detections: list[BlockDetection]):
    if block_detections or args.no_image_block_auto:
        return block_detections

    if args.box_bbox:
        box_bbox = clamp_bbox(parse_bbox(args.box_bbox), image.shape)
    else:
        box_bbox = detect_box_bbox(image)

    return detect_image_block_candidates(image, box_bbox)


def process_one_frame(
    image,
    args,
    yolo_model=None,
    depth_frame=None,
    intrinsics=None,
    rs_module=None,
):
    yolo_detections, annotated = detect_yolo_blocks(
        image=image,
        model=yolo_model,
        args=args,
        depth_frame=depth_frame,
        intrinsics=intrinsics,
        rs_module=rs_module,
    )
    block_detections = manual_block_detections(args, image.shape)
    block_detections.extend(yolo_detections)

    if args.image:
        block_detections = add_image_fallback_blocks(args, image, block_detections)

    if not args.enable_slot_grid:
        return None, annotated, block_detections

    manual_box_bbox = parse_bbox(args.box_bbox) if args.box_bbox else None
    result, debug_image = compute_slot_grid_result(
        image=image,
        block_detections=block_detections,
        rows=args.rows,
        cols=args.cols,
        box_bbox=manual_box_bbox,
    )
    return result, debug_image, block_detections


def run_image(args, yolo_model=None, ros_publisher=None):
    image_path = Path(args.image)
    image = read_image(image_path)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    if args.select_box_bbox:
        args.box_bbox = select_box_bbox_from_image(
            image,
            "Select Box BBox - Offline Image",
        )

    result, display_image, block_detections = process_one_frame(
        image=image,
        args=args,
        yolo_model=yolo_model,
    )

    print_block_summary(block_detections)

    if result is not None:
        print_slot_summary(result)
        json_path, debug_path = save_outputs(args, result, display_image, suffix=image_path.stem)
        maybe_publish_ros(ros_publisher, result)
        print(f"saved JSON: {json_path}")
        print(f"saved debug image: {debug_path}")
    else:
        debug_path = Path(args.output_dir) / f"{args.run_name}_{timestamp()}_{image_path.stem}_yolo_debug.jpg"
        write_image(debug_path, display_image)
        print(f"saved debug image: {debug_path}")

    if not args.no_show:
        cv2.imshow("D435/YOLO Offline Image + Slot Grid", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_camera(args, yolo_model=None, ros_publisher=None):
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is required for --camera. Install with: pip install -r requirements.txt"
        ) from exc

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    profile = pipeline.get_active_profile()
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsics = color_profile.get_intrinsics()

    last_save_time = 0.0
    print("RealSense D435 stream started.")
    print("Press 'q' or ESC in the OpenCV window to quit.")

    try:
        if args.select_box_bbox:
            print("Waiting for one D435 frame for box bbox selection...")
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                if not color_frame:
                    continue

                first_frame = np.asanyarray(color_frame.get_data())
                args.box_bbox = select_box_bbox_from_image(
                    first_frame,
                    "Select Box BBox - D435 Frame",
                )
                break

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            result, display_image, block_detections = process_one_frame(
                image=frame,
                args=args,
                yolo_model=yolo_model,
                depth_frame=depth_frame,
                intrinsics=intrinsics,
                rs_module=rs,
            )
            maybe_publish_ros(ros_publisher, result)

            now = time.time()
            if now - last_save_time >= args.save_interval:
                print_block_summary(block_detections)
                if result is not None:
                    print_slot_summary(result)
                    json_path, debug_path = save_outputs(args, result, display_image, suffix="camera")
                    print(f"saved JSON/debug image: {json_path}, {debug_path}")
                last_save_time = now

            if not args.no_show:
                cv2.imshow("D435/YOLO + Slot Grid", display_image)
                if args.show_depth:
                    cv2.imshow("Aligned Depth", make_depth_visualization(depth_frame))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
            else:
                time.sleep(0.01)

    finally:
        pipeline.stop()
        if not args.no_show:
            cv2.destroyAllWindows()


def main():
    args = parse_args()

    yolo_model = None
    if args.yolo_model:
        yolo_model_path = Path(args.yolo_model)
        if not yolo_model_path.exists():
            raise RuntimeError(f"YOLO model file not found: {yolo_model_path}")
        yolo_model = load_yolo_model(str(yolo_model_path))
        print(f"YOLO model loaded: {yolo_model_path}")
    else:
        print("YOLO model not provided. Running stream/manual/offline fallback mode.")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    ros_publisher = None
    try:
        if args.ros_publish:
            if not args.enable_slot_grid:
                print("Warning: --ros-publish is most useful with --enable-slot-grid.")
            ros_publisher = VisionRosPublisher(
                node_name=args.ros_node_name,
                camera_frame=args.camera_frame,
                publish_rate_hz=args.ros_rate,
            )

        if args.camera:
            run_camera(args, yolo_model=yolo_model, ros_publisher=ros_publisher)
        else:
            run_image(args, yolo_model=yolo_model, ros_publisher=ros_publisher)
    finally:
        if ros_publisher is not None:
            ros_publisher.shutdown()


if __name__ == "__main__":
    main()
