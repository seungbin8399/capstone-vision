# 연구실 D435 + YOLO + Slot Grid 실험 가이드

처음 보는 사람은 먼저 `BEGINNER_GUIDE.md`를 읽으면 전체 구조와 용어를 더 쉽게 이해할 수 있습니다. 이 파일은 연구실에서 바로 실행할 명령어 중심의 가이드입니다.

## 진짜 연구실에서 먼저 볼 명령어

### Windows PowerShell

환경 점검:

```powershell
python check_lab_ready.py --yolo-model best.pt --check-ros2
```

카메라 없이 offline/sample slot 흐름 확인:

```powershell
python realsense_xyz_yolo.py --image sample.jpg --enable-slot-grid --select-box-bbox --block-bbox 120,100,260,230 --no-show --save-latest
```

D435 + YOLO + slot grid 통합 실험:

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --select-box-bbox --show-depth --save-latest
```

### Ubuntu Bash

```bash
python3 check_lab_ready.py --yolo-model best.pt --check-ros2
python3 realsense_xyz_yolo.py --image sample.jpg --enable-slot-grid --select-box-bbox --block-bbox 120,100,260,230 --no-show --save-latest
python3 realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --select-box-bbox --show-depth --save-latest
```

`--select-box-bbox`를 쓰면 image 또는 첫 D435 frame에서 박스 영역을 마우스로 드래그합니다. 선택 후 터미널에 `--box-bbox x1,y1,x2,y2`가 출력되므로, 다음 실행부터 그 값을 복붙하면 됩니다.

## 이번 작업 범위

이번 단계는 비전 결과 확인까지만 합니다. 로봇팔 MoveIt execute, 실제 motion planning 실행, 강화학습 제어는 하지 않습니다.

출력 목표:

- 유압블럭 `bbox`, `center pixel`, `depth`, `camera XYZ`
- 박스 `bbox`
- slot별 `empty / occupied / target`
- target slot center pixel
- JSON 및 debug image 저장
- 추후 ROS2 `/target_slot_pixel`, `/target_slot_pose` 확장용 placeholder
- 선택 기능으로 ROS2 `/vision/...` topic publish

## 준비물

- Intel RealSense D435
- USB 3.x 케이블
- YOLO 모델 `best.pt`
- offline 테스트 이미지 `sample.jpg`
- 실행 PC
  - 현재 기준: Windows + PowerShell + Python
  - Ubuntu에서도 Python script 실행 가능
  - Ubuntu D435 사용 시 librealsense/RealSense SDK 상태 확인 필요

## 설치

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Ubuntu

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

필수 패키지:

- `pyrealsense2`
- `opencv-python`
- `numpy`
- `ultralytics`

## 실험 순서 5단계

### Step 1. 환경 점검

```powershell
python check_lab_ready.py --yolo-model best.pt
```

확인 항목:

- Python version
- OpenCV import
- pyrealsense2 import
- ultralytics import
- `best.pt` 존재 여부
- D435 연결 여부
- `outputs/` 생성 가능 여부
- sample image 존재 여부

### Step 2. D435 카메라 확인

```powershell
python realsense_xyz_yolo.py --camera --show-depth
```

RGB 창과 aligned depth 창이 보이면 카메라 입력은 정상입니다.

### Step 3. YOLO 유압블럭 인식 확인

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --show-depth
```

터미널에 block center, depth, camera XYZ가 출력되는지 확인합니다.

### Step 4. 박스 bbox 지정 또는 자동 검출

박스 bbox를 마우스로 지정:

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --select-box-bbox --show-depth --save-latest
```

수동 bbox를 이미 알고 있을 때:

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --box-bbox 120,90,540,420 --show-depth --save-latest
```

자동 박스 검출:

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --show-depth --save-latest
```

초기 연구실 실험에서는 자동보다 `--select-box-bbox` 또는 `--box-bbox`가 더 안정적입니다.

### Step 5. slot grid + JSON/debug 저장 확인

저장 위치:

```text
outputs/
```

`--save-latest` 사용 시 매번 확인하기 쉬운 파일도 같이 생성됩니다.

```text
outputs/lab_slot_grid_latest_result.json
outputs/lab_slot_grid_latest_debug.jpg
```

## 복붙용 명령어 모음

D435 연결/stream 확인:

```powershell
python realsense_xyz_yolo.py --camera --show-depth
```

YOLO block detection만 확인:

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --show-depth
```

YOLO + slot grid, 자동 박스 검출:

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --show-depth --save-latest
```

YOLO + slot grid, 마우스 박스 선택:

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --select-box-bbox --show-depth --save-latest
```

YOLO + slot grid, 수동 박스 bbox:

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --box-bbox 120,90,540,420 --show-depth --save-latest
```

offline image + 마우스 박스 선택 + 수동 block bbox:

```powershell
python realsense_xyz_yolo.py --image sample.jpg --enable-slot-grid --select-box-bbox --block-bbox 120,100,260,230 --no-show --save-latest
```

2x3 slot:

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --rows 2 --cols 3 --select-box-bbox --show-depth --save-latest
```

## 결과 JSON에서 꼭 볼 값

파일:

```text
outputs/lab_slot_grid_latest_result.json
```

핵심 필드:

- `blocks[0].center`: 유압블럭 center pixel
- `blocks[0].depth_m`: 유압블럭 depth
- `blocks[0].camera_xyz_m`: 유압블럭 camera frame XYZ
- `box.bbox`: 박스 bbox
- `slots`: slot bbox, center, status list
- `target_slot.slot_id`: target slot 번호
- `target_slot.center`: target slot center pixel
- `robot_target_placeholder`: 추후 ROS2/robot 좌표 변환용 placeholder

상태 규칙:

- 유압블럭 center가 slot 안에 있으면 `occupied`
- 유압블럭이 없는 slot은 `empty`
- 가장 번호가 빠른 empty slot이 `target`

## one-command 스크립트

Windows:

```powershell
.\run_lab_camera_test.bat
.\run_lab_slot_grid_test.bat
```

모델 경로가 다르면:

```powershell
.\run_lab_slot_grid_test.bat C:\path\to\best.pt
```

Ubuntu:

```bash
chmod +x run_lab_camera_test.sh run_lab_slot_grid_test.sh
./run_lab_camera_test.sh
./run_lab_slot_grid_test.sh
```

모델 경로가 다르면:

```bash
./run_lab_slot_grid_test.sh /path/to/best.pt
```

## 실패 시 짧은 체크리스트

### D435가 안 잡힐 때

- USB 3.x 포트에 연결했는지 확인
- RealSense Viewer가 켜져 있으면 닫기
- Windows 장치 관리자 또는 RealSense Viewer에서 D435 확인
- Ubuntu: `lsusb | grep -i realsense`

### `best.pt`가 없을 때

- repo root에 `best.pt`를 둡니다.
- 다른 경로면 `--yolo-model C:\path\to\best.pt`처럼 절대 경로를 넣습니다.

### YOLO는 되는데 박스 grid가 안 나올 때

- `--enable-slot-grid`를 넣었는지 확인
- `--save-latest`를 넣고 `outputs/lab_slot_grid_latest_debug.jpg` 확인
- 자동 검출 대신 `--select-box-bbox`로 박스를 직접 지정

### debug image는 저장됐는데 JSON이 비어 있을 때

- `outputs/lab_slot_grid_latest_result.json` 파일명을 확인
- `--enable-slot-grid` 없이 실행하면 slot JSON이 저장되지 않습니다.
- YOLO detection이 없으면 `blocks`가 비어 있을 수 있지만 target slot은 계산됩니다.

### 자동 박스 검출이 실패할 때

우회 1: 마우스로 선택

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --select-box-bbox --show-depth --save-latest
```

우회 2: 출력된 bbox를 다음 실행에 수동 입력

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --box-bbox x1,y1,x2,y2 --show-depth --save-latest
```

## ROS2 확장 TODO

현재는 JSON/debug image 저장까지만 합니다.

추후 확장 후보:

- `/target_slot_pixel`
- `/target_slot_pose`
- `/slot_occupancy`
- `/debug_image`

이번 실험에서는 robot arm execute를 호출하지 않습니다.

## ROS2 topic publish 옵션

ROS2 publish는 선택 기능입니다. `--ros-publish`를 켰을 때만 `rclpy`, `std_msgs`, `geometry_msgs`가 필요합니다. ROS2가 없는 PC에서도 일반 image/camera 테스트는 깨지지 않습니다.

ROS2 점검:

```powershell
python check_lab_ready.py --yolo-model best.pt --check-ros2
```

ROS2 topic publish 실행:

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --select-box-bbox --ros-publish --camera-frame camera_color_optical_frame --save-latest
```

publish topic:

- `/vision/slot_grid_result`
  - type: `std_msgs/String`
  - 내용: 현재 result JSON 전체 string
- `/vision/block_pose_camera`
  - type: `geometry_msgs/PoseStamped`
  - frame_id: 기본 `camera_color_optical_frame`
  - 내용: 유압블럭 중심 camera XYZ
  - orientation: identity quaternion
- `/vision/target_slot_pixel`
  - type: `geometry_msgs/PointStamped`
  - frame_id: 기본 `camera_color_optical_frame`
  - x: target slot center pixel x
  - y: target slot center pixel y
  - z: target slot id
- `/vision/target_slot_pose_camera`
  - type: `geometry_msgs/PoseStamped`
  - target slot depth/XYZ가 안정화되면 publish할 예정
  - 현재는 구조만 준비되어 있고, target slot camera XYZ가 없으면 publish하지 않음

토픽 확인:

```bash
ros2 topic list
ros2 topic echo /vision/slot_grid_result
ros2 topic echo /vision/block_pose_camera
ros2 topic echo /vision/target_slot_pixel
```

frame_id:

- 기본값: `camera_color_optical_frame`
- 변경 가능:

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --ros-publish --camera-frame camera_link --save-latest
```

중요 TODO:

- 지금 publish되는 pose는 camera frame 기준입니다.
- robot base 기준 좌표가 아닙니다.
- 추후 TF가 준비되면 `camera_color_optical_frame` -> `base_link` 또는 `link0` 변환을 tf2로 추가해야 합니다.
- 이번 코드에는 MoveIt planning/execute가 없습니다.
