import csv
from pathlib import Path
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


class CsvPosePublisher(Node):
    """Publish the last camera-frame 3D coordinate in a CSV as PoseStamped.

    Current meaning of the CSV columns:
    - center_u, center_v: candidate center pixel of the hydraulic block
    - sample_u, sample_v: pixel where depth was actually sampled
    - camera_x_m, camera_y_m, camera_z_m: 3D point in the RealSense camera frame

    This node does not transform to robot base_link yet.
    It publishes the pose in frame_id="camera_link" for the next ROS2 step.
    """

    def __init__(self):
        super().__init__("csv_pose_publisher")

        # Parameters make the node easier to reuse later without editing code.
        self.declare_parameter("csv_path", "realsense_depth_log_without_postit.csv")
        self.declare_parameter("frame_id", "camera_link")
        self.declare_parameter("topic_name", "/detected_object_pose")
        self.declare_parameter("publish_period_sec", 1.0)

        self.csv_path = self.get_parameter("csv_path").value
        self.frame_id = self.get_parameter("frame_id").value
        self.topic_name = self.get_parameter("topic_name").value
        self.publish_period_sec = float(
            self.get_parameter("publish_period_sec").value
        )

        self.pose_xyz = self.load_last_camera_xyz(self.csv_path)
        if self.pose_xyz is None:
            raise RuntimeError(
                f"Could not read a valid pose from CSV file: {self.csv_path}"
            )

        self.publisher = self.create_publisher(PoseStamped, self.topic_name, 10)
        self.timer = self.create_timer(self.publish_period_sec, self.publish_pose)

        x, y, z = self.pose_xyz
        self.get_logger().info(f"CSV path: {Path(self.csv_path).resolve()}")
        self.get_logger().info(f"Publishing PoseStamped to: {self.topic_name}")
        self.get_logger().info(f"frame_id: {self.frame_id}")
        self.get_logger().info(
            f"Loaded last camera pose: x={x:.6f}, y={y:.6f}, z={z:.6f} m"
        )

    def load_last_camera_xyz(self, csv_path: str) -> Optional[Tuple[float, float, float]]:
        """Read the last valid camera_x/y/z row from the CSV file."""
        path = Path(csv_path).expanduser()

        if not path.exists():
            self.get_logger().error(f"CSV file does not exist: {path}")
            return None

        last_valid_xyz = None

        with path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)

            required_columns = {
                "camera_x_m",
                "camera_y_m",
                "camera_z_m",
            }
            missing_columns = required_columns - set(reader.fieldnames or [])
            if missing_columns:
                self.get_logger().error(
                    f"CSV is missing columns: {sorted(missing_columns)}"
                )
                return None

            for row in reader:
                try:
                    x = float(row["camera_x_m"])
                    y = float(row["camera_y_m"])
                    z = float(row["camera_z_m"])
                except (TypeError, ValueError):
                    continue

                # z must be positive because it is depth in meters.
                if z <= 0.0:
                    continue

                last_valid_xyz = (x, y, z)

        return last_valid_xyz

    def publish_pose(self):
        """Publish the same loaded pose once every timer tick."""
        x, y, z = self.pose_xyz

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z

        # No object orientation is estimated yet.
        # Use identity quaternion as a safe default.
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        self.publisher.publish(msg)
        self.get_logger().info(
            f"Published pose: frame={self.frame_id}, "
            f"x={x:.6f}, y={y:.6f}, z={z:.6f} m"
        )


def main(args=None):
    rclpy.init(args=args)

    node = None
    try:
        node = CsvPosePublisher()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
