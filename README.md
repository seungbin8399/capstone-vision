# RealSense D435 Hydraulic Block Detector

Windows에서 Intel RealSense D435와 OpenCV만으로 유압블록을 먼저 검출하는 1단계 실험 코드입니다.

## 1. 목표

- RealSense D435에서 RGB/Depth 프레임 받기
- Depth 기반 mask 생성
- 가장 큰 contour를 유압블록 후보로 판단
- bounding box, 중심 pixel 좌표, 중심 depth 표시
- 나중에 ROS2 비전 노드로 옮기기 쉬운 구조 유지

## 2. 설치

PowerShell에서 프로젝트 폴더로 이동한 뒤 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

만약 `pyrealsense2` 설치가 실패하면 Python 버전 호환 문제일 수 있습니다. Windows에서는 Python 3.10 또는 3.11 환경을 권장합니다.

## 3. 파일 구조

```text
Capstone-vision/
  requirements.txt
  run_block_detector.py
  realsense_block_detector/
    __init__.py
    camera.py
    detector.py
```

- `camera.py`: RealSense RGB/Depth 입력 담당
- `detector.py`: OpenCV depth mask, contour, bbox, center, depth 계산 담당
- `run_block_detector.py`: 실행 파일, 화면 표시, 터미널 출력 담당

## 4. 실행

RealSense Viewer를 닫은 뒤 실행합니다. Viewer가 카메라를 잡고 있으면 Python에서 카메라를 열 수 없습니다.

```powershell
.\.venv\Scripts\Activate.ps1
python run_block_detector.py
```

거리 범위를 조절하고 싶으면 다음처럼 실행합니다.

```powershell
python run_block_detector.py --min-depth 0.20 --max-depth 0.80 --min-area 2000
```

## 5. 실험 방법

1. 책상 위 배경을 최대한 단순하게 둡니다.
2. 유압블록을 카메라 앞에 놓습니다.
3. 유압블록이 배경보다 카메라에 더 가깝게 보이도록 합니다.
4. `Depth Range Mask` 창에서 유압블록이 흰색으로 잘 잡히는지 봅니다.
5. 배경까지 흰색으로 많이 잡히면 `--max-depth` 값을 줄입니다.
6. 작은 노이즈가 잡히면 `--min-area` 값을 키웁니다.

## 6. ROS2 노드로 옮길 때 publish할 값

나중에 ROS2로 옮길 때는 `detector.detect(...)` 결과인 `Detection` 값을 메시지로 publish하면 됩니다.

처음에는 다음 값을 publish하면 충분합니다.

- `center_u`: 중심 pixel x 좌표
- `center_v`: 중심 pixel y 좌표
- `depth_m`: 중심 depth, meter 단위
- `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`: bounding box
- `detected`: 검출 성공 여부

그 다음 단계에서는 RealSense camera intrinsics를 이용해 `(u, v, depth)`를 카메라 좌표계의 `(X, Y, Z)`로 바꿉니다.

```text
pixel center + depth
  -> camera frame 3D point
  -> robot base frame point
  -> motion planning target pose
```

ROS2에서는 보통 다음처럼 나눕니다.

- subscribe: `/camera/color/image_raw`
- subscribe: `/camera/aligned_depth_to_color/image_raw`
- publish: `/block_detection`
- publish: `/block_center_point`
- optional publish: `/debug_image`

## 7. 다음 단계

이 코드가 안정적으로 동작하면 다음 순서로 확장하면 좋습니다.

1. RealSense intrinsics로 중심점을 3D 좌표로 변환
2. 카메라 좌표계에서 로봇 베이스 좌표계로 extrinsic transform 적용
3. 유압블록의 방향 추정 추가
4. ROS2 `rclpy` 노드로 포팅
5. 목표 슬롯 근처에서만 강화학습 기반 미세 보정 적용
