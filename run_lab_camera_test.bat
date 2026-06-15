@echo off
setlocal

cd /d "%~dp0"
if not exist outputs mkdir outputs

python check_lab_ready.py --output-dir outputs
python realsense_xyz_yolo.py --camera --show-depth --output-dir outputs --run-name lab_camera_stream

endlocal
