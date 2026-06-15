# 초보자용 비전 시스템 설명서

이 문서는 지금까지 만든 D435 + YOLO + slot grid 비전 코드를 처음 보는 사람도 이해할 수 있도록 설명한 가이드입니다.

연구실에서 바로 실행할 명령어만 필요하면 `README_LAB_TEST.md`를 보면 됩니다.  
이 문서는 “왜 이런 구조인지”, “각 파일이 무슨 역할인지”, “결과 JSON과 ROS2 topic이 무엇을 의미하는지”를 천천히 설명합니다.

## 1. 이 프로젝트가 하려는 일

우리 프로젝트의 큰 목표는 다음과 같습니다.

```text
유압블럭을 카메라로 본다
-> 적재 박스를 찾는다
-> 박스 안을 여러 칸(slot)으로 나눈다
-> 어느 칸이 비었는지 판단한다
-> 비어 있는 칸 하나를 target slot으로 고른다
-> 그 결과를 JSON 또는 ROS2 topic으로 내보낸다
```

중요한 점:

- 이번 코드는 로봇팔을 움직이지 않습니다.
- MoveIt planning/execute를 호출하지 않습니다.
- 강화학습 제어도 하지 않습니다.
- 지금 목표는 비전 결과를 안정적으로 만들고 저장/publish하는 것입니다.

즉, 현재 단계는 로봇팔에게 “어디로 가야 하는지 알려줄 준비”까지만 합니다.

## 2. 전체 실행 흐름

메인 실행 파일은 `realsense_xyz_yolo.py`입니다.

실행 흐름은 다음과 같습니다.

```text
1. D435 카메라에서 RGB/depth frame 받기
2. YOLO 모델로 유압블럭 bbox 찾기
3. bbox 중심 pixel 계산
4. 중심 pixel에서 depth 읽기
5. depth와 카메라 intrinsics로 camera XYZ 계산
6. 같은 RGB frame에서 적재 박스 bbox 찾기
7. 박스 내부를 rows/cols 기반 slot으로 나누기
8. 유압블럭 중심이 들어간 slot을 occupied로 표시
9. 비어 있는 slot 중 가장 빠른 번호를 target으로 선택
10. JSON/debug image 저장
11. 선택적으로 ROS2 topic publish
```

## 3. 꼭 알아야 할 용어

### D435

Intel RealSense D435 카메라입니다. RGB 영상과 depth 영상을 같이 줍니다.

- RGB: 일반 카메라 이미지
- Depth: 각 pixel까지의 거리

### YOLO

이미지에서 물체를 찾는 딥러닝 모델입니다. 여기서는 유압블럭을 찾는 데 사용합니다.

YOLO 결과는 보통 이런 값을 줍니다.

- bbox
- class label
- confidence

### bbox

bounding box의 줄임말입니다. 물체를 감싸는 사각형입니다.

형식:

```text
x1,y1,x2,y2
```

예시:

```text
120,90,540,420
```

뜻:

- 왼쪽 위 점: `(120, 90)`
- 오른쪽 아래 점: `(540, 420)`

### center pixel

bbox의 중심 pixel입니다.

예를 들어 bbox가 `120,90,540,420`이면 중심은 대략:

```text
((120+540)/2, (90+420)/2) = (330, 255)
```

### camera XYZ

D435 카메라 기준 3D 좌표입니다.

주의:

- robot base 좌표가 아닙니다.
- `camera_color_optical_frame` 기준입니다.
- 나중에 tf2로 robot `base_link` 또는 `link0` 기준으로 변환해야 합니다.

### slot

박스 내부를 나눈 칸입니다.

기본값은 2행 2열입니다.

```text
slot 1 | slot 2
-------+-------
slot 3 | slot 4
```

### occupied / empty / target

slot 상태입니다.

- `occupied`: 유압블럭이 들어 있는 칸
- `empty`: 비어 있는 칸
- `target`: 비어 있는 칸 중 이번에 적재 목표로 고른 칸

현재 target 선택 규칙은 단순합니다.

```text
가장 번호가 빠른 empty slot을 target으로 선택
```

## 4. 주요 파일 설명

### `realsense_xyz_yolo.py`

가장 중요한 메인 실행 파일입니다.

하는 일:

- D435 camera 실행
- image offline 테스트
- YOLO 유압블럭 인식
- block bbox/center/depth/camera XYZ 계산
- slot grid 계산
- JSON/debug image 저장
- 선택적으로 ROS2 topic publish

가장 많이 쓰는 명령어:

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --select-box-bbox --show-depth --save-latest
```

### `realsense_block_detector/slot_grid.py`

slot/grid 계산을 담당하는 파일입니다.

하는 일:

- YOLO 결과를 공통 detection 형식으로 바꾸기
- 박스 bbox 찾기
- 박스 내부 slot 나누기
- occupied/empty/target 판단
- debug image 그리기
- JSON 결과 만들기

### `check_lab_ready.py`

연구실 실험 전에 환경이 준비됐는지 확인하는 파일입니다.

확인하는 것:

- Python version
- OpenCV import
- pyrealsense2 import
- ultralytics import
- best.pt 존재 여부
- D435 연결 여부
- outputs 폴더 생성 가능 여부
- sample image 존재 여부
- 선택적으로 ROS2 import

실행:

```powershell
python check_lab_ready.py --yolo-model best.pt --check-ros2
```

### `README_LAB_TEST.md`

연구실에서 바로 보고 따라 할 실행 가이드입니다.

초보자가 개념을 이해하려면 이 문서를 먼저 보고, 실제 실험할 때는 `README_LAB_TEST.md`의 명령어를 보면 됩니다.

## 5. 처음 실행하는 순서

### Step 1. 환경 확인

```powershell
python check_lab_ready.py --yolo-model best.pt --check-ros2
```

결과 예시:

```text
[OK] Python version
[OK] OpenCV import
[OK] pyrealsense2 import
[OK] NumPy import
[OK] Ultralytics import
[OK] YOLO model found
[OK] outputs directory ready
[OK] D435 camera detected
```

`[FAIL]`이 있으면 full camera + YOLO 실험 전에 해결해야 합니다.

`[WARN] D435 camera detected`는 카메라가 연결되지 않았다는 뜻입니다. 이 경우 offline image 테스트는 할 수 있습니다.

### Step 2. 카메라 없이 offline 테스트

sample image가 있을 때:

```powershell
python realsense_xyz_yolo.py --image sample.jpg --enable-slot-grid --select-box-bbox --block-bbox 120,100,260,230 --no-show --save-latest
```

이 명령은 실제 D435 없이도 slot grid 흐름을 확인하기 위한 것입니다.

`--block-bbox`는 임시로 유압블럭 위치를 직접 넣는 옵션입니다.

### Step 3. D435 stream 확인

```powershell
python realsense_xyz_yolo.py --camera --show-depth
```

확인할 것:

- RGB 화면이 나오는지
- depth 화면이 나오는지
- 화면이 멈추지 않는지

### Step 4. YOLO 유압블럭 인식 확인

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --show-depth
```

터미널에 이런 값이 나와야 합니다.

```text
block label=... conf=... center=(x,y) depth=... xyz=(X,Y,Z)
```

### Step 5. slot grid까지 확인

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --select-box-bbox --show-depth --save-latest
```

실행하면 첫 D435 frame이 뜨고, 마우스로 적재 박스 영역을 드래그합니다.

드래그 후 터미널에 이런 값이 출력됩니다.

```text
Selected box bbox: --box-bbox 120,90,540,420
```

다음부터는 이 값을 복사해서 수동 bbox로 실행할 수 있습니다.

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --box-bbox 120,90,540,420 --show-depth --save-latest
```

## 6. 명령어 옵션 설명

### `--camera`

D435 카메라를 사용합니다.

```powershell
python realsense_xyz_yolo.py --camera
```

### `--image sample.jpg`

카메라 대신 이미지 파일을 사용합니다.

```powershell
python realsense_xyz_yolo.py --image sample.jpg
```

### `--yolo-model best.pt`

YOLO 모델 파일을 사용합니다.

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt
```

### `--enable-slot-grid`

박스 grid/slot 계산을 켭니다.

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid
```

### `--select-box-bbox`

마우스로 박스 bbox를 선택합니다.

초기 연구실 실험에서 가장 추천하는 방식입니다.

### `--box-bbox x1,y1,x2,y2`

박스 bbox를 직접 입력합니다.

자동 박스 검출이 흔들릴 때 사용합니다.

### `--block-bbox x1,y1,x2,y2`

유압블럭 bbox를 직접 입력합니다.

카메라나 YOLO 없이 slot 로직만 테스트할 때 사용합니다.

### `--save-latest`

항상 같은 최신 파일명으로 결과를 저장합니다.

확인하기 쉬운 파일:

```text
outputs/lab_slot_grid_latest_result.json
outputs/lab_slot_grid_latest_debug.jpg
```

### `--ros-publish`

ROS2 topic publish를 켭니다.

ROS2가 설치된 환경에서만 사용합니다.

## 7. 결과 파일 설명

결과는 `outputs/` 폴더에 저장됩니다.

대표 파일:

```text
outputs/lab_slot_grid_latest_result.json
outputs/lab_slot_grid_latest_debug.jpg
```

### debug image

이미지 위에 다음이 그려집니다.

- 박스 외곽선
- slot 경계선
- slot 번호
- `occupied / empty / target`
- 유압블럭 bbox
- 유압블럭 center

### JSON

중요 필드는 다음입니다.

```json
{
  "box": {
    "bbox": [0, 0, 640, 480],
    "center": [320, 240]
  },
  "slots": [
    {
      "slot_id": 1,
      "bbox": [0, 0, 320, 240],
      "center": [160, 120],
      "status": "occupied"
    }
  ],
  "target_slot": {
    "slot_id": 2,
    "center": [480, 120]
  },
  "blocks": [
    {
      "bbox": [100, 80, 240, 210],
      "center": [170, 145],
      "depth_m": 0.42,
      "camera_xyz_m": [-0.05, 0.03, 0.42]
    }
  ]
}
```

꼭 봐야 할 값:

- `blocks[0].center`: 유압블럭 중심 pixel
- `blocks[0].depth_m`: 유압블럭 depth
- `blocks[0].camera_xyz_m`: 카메라 기준 3D 좌표
- `box.bbox`: 적재 박스 bbox
- `slots`: 각 slot 상태
- `target_slot.center`: target slot 중심 pixel

## 8. ROS2 topic publish

ROS2는 선택 기능입니다.

일반 테스트에서는 필요 없습니다.

ROS2 publish 실행:

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --select-box-bbox --ros-publish --camera-frame camera_color_optical_frame --save-latest
```

publish되는 topic:

```text
/vision/slot_grid_result
/vision/block_pose_camera
/vision/target_slot_pixel
```

확인:

```bash
ros2 topic list
ros2 topic echo /vision/slot_grid_result
ros2 topic echo /vision/block_pose_camera
ros2 topic echo /vision/target_slot_pixel
```

주의:

- `/vision/block_pose_camera`는 camera frame 기준입니다.
- robot base frame 기준이 아닙니다.
- 나중에 tf2로 `camera_color_optical_frame`에서 `base_link` 또는 `link0`로 변환해야 합니다.
- 현재 코드에는 MoveIt execute가 없습니다.

## 9. 자주 생기는 문제

### D435가 안 잡힘

확인:

- USB 3.x 포트인지 확인
- RealSense Viewer가 켜져 있으면 닫기
- 케이블 다시 연결
- Windows 장치 관리자 확인
- Ubuntu라면 `lsusb | grep -i realsense`

### `best.pt`가 없다고 나옴

`best.pt`를 repo root에 둡니다.

다른 폴더에 있다면 경로를 직접 입력합니다.

```powershell
python realsense_xyz_yolo.py --camera --yolo-model C:\path\to\best.pt
```

### YOLO는 되는데 slot grid가 이상함

대부분 박스 bbox가 잘못 잡힌 경우입니다.

해결:

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --select-box-bbox --show-depth --save-latest
```

또는 직접 입력:

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --box-bbox 120,90,540,420 --show-depth --save-latest
```

### JSON은 있는데 block 정보가 비어 있음

YOLO가 유압블럭을 못 찾은 것입니다.

확인:

- `best.pt`가 맞는 모델인지
- 유압블럭이 화면에 잘 보이는지
- 조명이 너무 어둡거나 반사가 심하지 않은지

### ROS2 topic이 안 보임

확인:

```bash
ros2 topic list
echo $ROS_DOMAIN_ID
```

Windows PowerShell:

```powershell
echo $env:ROS_DOMAIN_ID
```

`--ros-publish` 옵션을 넣었는지도 확인합니다.

## 10. 초보자용 추천 실행 3개

처음 연구실에 가면 이 순서대로 실행하세요.

```powershell
python check_lab_ready.py --yolo-model best.pt --check-ros2
python realsense_xyz_yolo.py --camera --show-depth
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --select-box-bbox --show-depth --save-latest
```

ROS2 topic까지 확인하려면 마지막 명령을 이렇게 바꿉니다.

```powershell
python realsense_xyz_yolo.py --camera --yolo-model best.pt --enable-slot-grid --select-box-bbox --ros-publish --camera-frame camera_color_optical_frame --show-depth --save-latest
```
