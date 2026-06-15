#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p outputs

python3 check_lab_ready.py --output-dir outputs
python3 realsense_xyz_yolo.py --camera --show-depth --output-dir outputs --run-name lab_camera_stream
