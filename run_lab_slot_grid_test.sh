#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p outputs

YOLO_MODEL="${1:-best.pt}"

python3 check_lab_ready.py --yolo-model "$YOLO_MODEL" --output-dir outputs
python3 realsense_xyz_yolo.py --camera --yolo-model "$YOLO_MODEL" --enable-slot-grid --select-box-bbox --show-depth --save-latest --output-dir outputs --run-name lab_slot_grid
