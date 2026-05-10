import pyrealsense2 as rs
import numpy as np


class RealSenseCamera:
    """Small wrapper around Intel RealSense color/depth streaming.

    Keeping camera code in one class makes it easier to replace this file with
    ROS2 image subscribers later.
    """

    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.depth_scale = None
        self.started = False

    def start(self):
        """Start color and depth streams."""
        self.config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )
        self.config.enable_stream(
            rs.stream.depth,
            self.width,
            self.height,
            rs.format.z16,
            self.fps,
        )

        profile = self.pipeline.start(self.config)
        self.started = True
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        return self.depth_scale

    def get_frames(self):
        """Return aligned color image, depth image, and RealSense depth frame.

        color_image: OpenCV BGR image, shape=(H, W, 3)
        depth_image: raw uint16 depth image, shape=(H, W)
        depth_frame: RealSense frame object used for get_distance(u, v)
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image, depth_frame

    def stop(self):
        """Stop the RealSense pipeline."""
        if self.started:
            self.pipeline.stop()
            self.started = False
