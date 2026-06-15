import argparse
import os
import platform
import sys
from pathlib import Path


ISSUES = {"FAIL": 0, "WARN": 0}


def print_status(level: str, label: str, detail: str = "") -> None:
    if level in ISSUES:
        ISSUES[level] += 1
    message = f"[{level}] {label}"
    if detail:
        message += f" - {detail}"
    print(message)


def import_check(module_name: str, label: str, required: bool = True):
    try:
        module = __import__(module_name)
    except Exception as exc:
        print_status("FAIL" if required else "WARN", label, str(exc))
        return None

    version = getattr(module, "__version__", None)
    detail = f"version {version}" if version else "imported"
    print_status("OK", label, detail)
    return module


def find_sample_image(requested: str | None) -> Path | None:
    candidates = []
    if requested:
        candidates.append(Path(requested))

    candidates.extend(
        [
            Path("sample.jpg"),
            Path("sample.png"),
            Path("experiment_images_contact_sheet.png"),
        ]
    )

    for path in candidates:
        if path.exists():
            return path
    return None


def check_realsense_device(rs_module) -> bool:
    if rs_module is None:
        print_status("WARN", "D435 camera detected", "skipped because pyrealsense2 import failed")
        return False

    try:
        context = rs_module.context()
        devices = context.query_devices()
    except Exception as exc:
        print_status("WARN", "D435 camera detected", str(exc))
        return False

    if len(devices) <= 0:
        print_status("WARN", "D435 camera detected", "no RealSense device found")
        return False

    names = []
    for device in devices:
        try:
            names.append(device.get_info(rs_module.camera_info.name))
        except Exception:
            names.append("RealSense device")

    print_status("OK", "D435 camera detected", ", ".join(names))
    return True


def check_output_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        print_status("FAIL", "outputs directory ready", str(exc))
        return False

    print_status("OK", "outputs directory ready", str(path.resolve()))
    return True


def main():
    parser = argparse.ArgumentParser(description="Check lab readiness for D435 slot-grid vision tests.")
    parser.add_argument("--yolo-model", default="best.pt")
    parser.add_argument("--sample-image")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--check-ros2",
        action="store_true",
        help="Also check ROS2 Python packages for optional topic publishing.",
    )
    args = parser.parse_args()

    print("Lab readiness check")
    print("===================")

    py_ok = sys.version_info >= (3, 9)
    print_status(
        "OK" if py_ok else "FAIL",
        "Python version",
        f"{platform.python_version()} ({platform.system()} {platform.release()})",
    )

    cv2_module = import_check("cv2", "OpenCV import", required=True)
    rs_module = import_check("pyrealsense2", "pyrealsense2 import", required=True)
    import_check("numpy", "NumPy import", required=True)
    import_check("ultralytics", "Ultralytics import", required=True)

    model_path = Path(args.yolo_model)
    print_status(
        "OK" if model_path.exists() else "FAIL",
        "YOLO model found",
        str(model_path),
    )

    sample_path = find_sample_image(args.sample_image)
    print_status(
        "OK" if sample_path is not None else "WARN",
        "sample image found",
        str(sample_path) if sample_path else "not found",
    )

    check_output_dir(Path(args.output_dir))
    camera_ok = check_realsense_device(rs_module)

    if args.check_ros2:
        print()
        print("ROS2 optional publish check")
        print("---------------------------")
        import_check("rclpy", "rclpy import", required=False)
        import_check("std_msgs.msg", "std_msgs import", required=False)
        import_check("geometry_msgs.msg", "geometry_msgs import", required=False)
        print_status("OK", "ROS_DOMAIN_ID", os.environ.get("ROS_DOMAIN_ID", "not set"))

    print()
    print("Summary")
    print("-------")
    if ISSUES["FAIL"] == 0 and ISSUES["WARN"] == 0:
        print("[OK] Lab environment is ready.")
    else:
        print(f"[INFO] FAIL={ISSUES['FAIL']} WARN={ISSUES['WARN']}")
        if ISSUES["FAIL"] > 0:
            print("[INFO] Fix FAIL items before full D435 + YOLO testing.")
        if not camera_ok:
            print("[INFO] D435 is optional for offline fallback, but required for lab camera testing.")

    print()
    print("Suggested next commands")
    print("-----------------------")
    if sample_path is not None:
        print(
            "Offline/sample test: "
            f"python realsense_xyz_yolo.py --image \"{sample_path}\" "
            "--enable-slot-grid --select-box-bbox --block-bbox 10,10,120,120 --no-show --save-latest"
        )
    if camera_ok:
        print("D435 stream check: python realsense_xyz_yolo.py --camera --show-depth")
        if model_path.exists():
            print(
                "D435 + YOLO + slot grid: "
                f"python realsense_xyz_yolo.py --camera --yolo-model \"{model_path}\" "
                "--enable-slot-grid --select-box-bbox --show-depth --save-latest"
            )


if __name__ == "__main__":
    main()
