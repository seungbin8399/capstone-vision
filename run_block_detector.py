import argparse
import time

import cv2

from realsense_block_detector.camera import RealSenseCamera
from realsense_block_detector.detector import (
    HydraulicBlockDetector,
    draw_detection,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect a hydraulic block using Intel RealSense D435 depth."
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.15,
        help="Ignore pixels closer than this distance in meters.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=1.20,
        help="Ignore pixels farther than this distance in meters.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=1200,
        help="Ignore contours smaller than this pixel area.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    camera = RealSenseCamera(width=args.width, height=args.height, fps=args.fps)
    detector = HydraulicBlockDetector(
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        min_area_px=args.min_area,
    )

    try:
        depth_scale = camera.start()
        print(f"RealSense started. depth_scale={depth_scale}")
        print("Press 'q' or ESC in the OpenCV window to quit.")

        last_print_time = 0.0

        while True:
            color_image, depth_image, depth_frame = camera.get_frames()
            if color_image is None:
                continue

            detection, mask = detector.detect(
                color_image=color_image,
                depth_image=depth_image,
                depth_scale=depth_scale,
                depth_frame=depth_frame,
            )

            result_image = draw_detection(color_image, detection)

            # Show both the final result and the binary depth mask used by OpenCV.
            cv2.imshow("Hydraulic Block Detection", result_image)
            cv2.imshow("Depth Range Mask", mask)

            # Print at a readable rate instead of flooding the terminal.
            now = time.time()
            if detection is not None and now - last_print_time > 0.5:
                u, v = detection.center
                print(
                    f"center=({u}, {v}), depth={detection.depth_m:.3f}m, "
                    f"bbox={detection.bbox}, area={detection.area:.1f}"
                )
                last_print_time = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    except RuntimeError as exc:
        print("RealSense error:")
        print(exc)
        print()
        print("Check that:")
        print("1. The D435 is connected by USB 3.x.")
        print("2. RealSense Viewer is closed before running this script.")
        print("3. Color Camera and Stereo Module work in RealSense Viewer.")
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
