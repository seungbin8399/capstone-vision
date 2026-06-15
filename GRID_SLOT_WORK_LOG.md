# Grid/Slot Occupancy MVP 작업 로그

## 목표

Intel RealSense D435 RGB/Depth 입력에서 적재 박스와 유압블럭 위치를 인식하고, 박스 내부를 rows/cols 기반 slot으로 나눈 뒤 각 slot 상태를 `empty`, `occupied`, `target`으로 판단한다.

현재 MVP는 강화학습이 아니라 비전 파트에 집중한다. 큰 로봇 이동은 MoveIt/Motion Planning/rule-based로 처리하고, 이 비전 결과는 추후 ROS2 topic 또는 JSON으로 전달할 수 있는 형태를 목표로 한다.

## 로컬 프로젝트 분석

분석한 주요 파일:

- `realsense_block_detector/camera.py`
  - Intel RealSense D435 color/depth stream 시작
  - depth frame을 color frame에 align
  - OpenCV BGR 이미지, raw depth image, RealSense depth_frame 반환

- `realsense_block_detector/detector.py`
  - depth threshold 기반 mask 생성
  - 가장 큰 contour를 유압블럭 후보로 선택
  - 출력: `bbox`, `center`, `area`, `depth_m`
  - bbox는 `(x, y, w, h)` 형식

- `run_block_detector.py`
  - 기존 D435 유압블럭 검출 실행 스크립트
  - `HydraulicBlockDetector` 결과를 화면에 표시
  - 기존 실험 코드는 유지

- `realsense_block_detect.py`
  - 긴 실험용 스크립트
  - fixed ROI, contour mode, depth median, camera XYZ 계산, CSV logging 포함
  - 과거 실험 파라미터가 많아 MVP에서는 직접 수정하지 않고 참고 코드로 유지

- `vision_module/vision_module/csv_pose_publisher.py`
  - CSV의 마지막 camera XYZ 값을 ROS2 `PoseStamped`로 publish
  - 추후 slot target center 또는 target pose publish로 확장 가능

살릴 부분:

- D435 stream wrapper: `RealSenseCamera`
- depth 기반 유압블럭 후보 검출: `HydraulicBlockDetector`
- camera XYZ 계산 아이디어: `realsense_block_detect.py`, 친구 repo의 `rs2_deproject_pixel_to_point`
- ROS2 확장 방향: `csv_pose_publisher.py`

보완한 부분:

- 박스 외곽 bbox 검출
- rows/cols 기반 slot 분할
- block bbox/center와 slot 연결
- empty/occupied/target 상태 계산
- JSON 결과 저장
- 발표용 debug image 저장

## 친구 repo 분석

참고 repo:

`https://github.com/sukhee04/capstone-vision-system`

확인한 파일:

- `realsense_xyz_yolo.py`
- `wrist_camera_yolo.py`
- `realsense_custom.py`
- `realsense_roboflow.py`
- `local_yolo.py`
- `webcam_yolo.py`

핵심 통합 단위:

- YOLO 결과 bbox: `box.xyxy[0]` -> `[x1, y1, x2, y2]`
- 중심 pixel: `cx = (x1 + x2) / 2`, `cy = (y1 + y2) / 2`
- depth: `depth_frame.get_distance(cx, cy)`
- camera coordinate: `rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)`
- 출력 가능 값: bbox, center pixel, confidence, class name, depth, camera XYZ

mask 출력은 친구 repo의 기본 YOLO 흐름에는 없다. 따라서 현재 MVP에서는 bbox/center 기반으로 slot occupancy를 판단한다.

## 새로 추가한 파일

- `realsense_block_detector/slot_grid.py`
  - box bbox 자동 검출
  - grid/slot 생성
  - 유압블럭 center가 포함된 slot을 `occupied`로 표시
  - 가장 번호가 빠른 empty slot을 `target`으로 선택
  - JSON 결과 생성
  - debug image drawing

- `run_grid_box_detection.py`
  - 이미지 파일 테스트
  - D435 실시간 카메라 테스트
  - 수동 bbox 입력 지원
  - YOLO model 선택 지원
  - 기존 `HydraulicBlockDetector` depth contour fallback 지원

## 실행 방법

이미지 파일 테스트:

```powershell
python run_grid_box_detection.py --image sample.jpg
```

2x2가 아닌 grid:

```powershell
python run_grid_box_detection.py --image sample.jpg --rows 2 --cols 3
```

박스 bbox를 수동으로 지정:

```powershell
python run_grid_box_detection.py --image sample.jpg --box-bbox 100,80,540,430
```

유압블럭 bbox를 수동으로 지정:

```powershell
python run_grid_box_detection.py --image sample.jpg --block-bbox 120,100,260,230
```

친구 repo 방식의 YOLO model을 이용:

```powershell
python run_grid_box_detection.py --image sample.jpg --yolo-model best.pt
```

D435 실시간 테스트:

```powershell
python run_grid_box_detection.py --camera
```

D435 + YOLO model:

```powershell
python run_grid_box_detection.py --camera --yolo-model best.pt
```

카메라 preview 창 없이 JSON/debug image만 갱신:

```powershell
python run_grid_box_detection.py --camera --no-show
```

## 출력 파일

기본 출력 위치:

- `outputs/<image_name>_grid_result.json`
- `outputs/<image_name>_grid_debug.jpg`
- camera mode에서는 `outputs/camera_grid_result.json`, `outputs/camera_grid_debug.jpg`

JSON 예시:

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
    },
    {
      "slot_id": 2,
      "bbox": [320, 0, 640, 240],
      "center": [480, 120],
      "status": "target"
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
      "label": "block",
      "confidence": 0.91,
      "depth_m": 0.42,
      "camera_xyz_m": [-0.05, 0.03, 0.42]
    }
  ]
}
```

## 현재 판단 규칙

1. 박스 bbox를 찾는다.
2. bbox를 rows/cols로 row-major 순서의 slot으로 나눈다.
3. 유압블럭 detection center가 slot bbox 안에 있으면 해당 slot은 `occupied`.
4. 유압블럭 center가 없는 slot은 `empty`.
5. `slot_id`가 가장 작은 empty slot 하나를 `target`으로 바꾼다.

## ROS2 topic 확장 방향

현재 JSON의 `target_slot.center`는 pixel 좌표다. 실제 로봇팔 적재에는 다음 단계가 필요하다.

1. target slot center pixel과 slot 주변 depth를 읽는다.
2. RealSense intrinsics로 pixel/depth를 camera frame XYZ로 변환한다.
3. TF 또는 extrinsic calibration으로 `camera_link` 좌표를 robot `base_link` 좌표로 변환한다.
4. ROS2 message로 publish한다.

추천 topic:

- `/vision/slot_occupancy`
  - 전체 JSON과 유사한 slot 상태 배열
- `/vision/target_slot`
  - target slot id, center pixel, camera XYZ
- `/vision/debug_image`
  - 박스/slot/상태가 그려진 image

기존 `CsvPosePublisher`는 CSV 기반 pose publish 예제이므로, 다음 단계에서는 JSON 또는 실시간 detection 결과를 읽는 `SlotTargetPublisher`로 확장하면 된다.
