@echo off
setlocal

cd /d "%~dp0"
if not exist outputs mkdir outputs

set YOLO_MODEL=best.pt
if not "%~1"=="" set YOLO_MODEL=%~1

python check_lab_ready.py --yolo-model "%YOLO_MODEL%" --output-dir outputs
python realsense_xyz_yolo.py --camera --yolo-model "%YOLO_MODEL%" --enable-slot-grid --select-box-bbox --show-depth --save-latest --output-dir outputs --run-name lab_slot_grid

endlocal
