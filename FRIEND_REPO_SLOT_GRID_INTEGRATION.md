# 친구 D435/YOLO 코드 기준 Slot Grid 통합 정리

## 기준 변경

최종 기준은 로컬 MVP 실행 파일이 아니라 친구 repo의 D435 유압블럭 인식 흐름이다.

참고 repo:

`https://github.com/sukhee04/capstone-vision-system`

메인 통합 대상:

- `realsense_xyz_yolo.py`

이 파일은 다음 핵심 값을 이미 만든다.

- D435 color frame
- aligned depth frame
- YOLO bbox: `box.xyxy[0]`
- center pixel: `(cx, cy)`
- depth: `depth_frame.get_distance(cx, cy)`
- camera XYZ: `rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)`

따라서 slot/grid 기능은 이 흐름 뒤에 붙는 post-processing이어야 한다.

## 친구 repo 핵심 파일 분석

- `realsense_xyz_yolo.py`
  - D435 color/depth stream 실행
  - depth를 color frame에 align
  - YOLO로 유압블럭 bbox 검출
  - bbox center pixel 계산
  - center depth 읽기
  - camera XYZ 계산
  - OpenCV 창에 결과 표시

- `wrist_camera_yolo.py`
  - `realsense_xyz_yolo.py`와 거의 같은 구조
  - wrist camera 실험용 이름이지만 D435/YOLO/XYZ 흐름은 동일

- `realsense_custom.py`
  - D435 + YOLO + center/depth 표시
  - XYZ 변환은 `realsense_xyz_yolo.py`보다 덜 정리되어 있음

최종 통합 기준은 `bbox`, `center`, `depth`, `camera XYZ`가 모두 있는 `realsense_xyz_yolo.py`가 가장 적합하다.

## 이번에 분리한 재사용 함수

파일:

- `realsense_block_detector/slot_grid.py`

재사용 대상:

- `BlockDetection`
  - YOLO 결과를 slot matching에 쓰기 위한 공통 detection 구조

- `yolo_results_to_block_detections(...)`
  - 친구 repo YOLO 결과를 `BlockDetection` list로 변환
  - bbox, center, confidence, depth, camera XYZ 포함

- `detect_box_bbox(...)`
  - 같은 D435 RGB frame에서 박스 외곽 bbox 추정

- `build_slots(...)`
  - rows/cols 기반 slot bbox/center 생성

- `assign_slot_statuses(...)`
  - block center가 들어간 slot을 `occupied`로 표시
  - 나머지는 `empty`
  - 가장 빠른 empty slot을 `target`으로 선택

- `compute_slot_grid_result(...)`
  - 한 frame 기준 통합 adapter
  - 입력: D435 color frame, YOLO block detections, rows, cols, optional box bbox
  - 출력: JSON dict, debug image

- `draw_debug_image(...)`
  - 박스 외곽선, slot 경계, slot_id/status, 유압블럭 bbox/center 표시

- `save_result_json(...)`
  - 결과 JSON 저장

## 친구 repo 흐름에 맞춘 최종 실행 파일

파일:

- `realsense_xyz_yolo.py`

이 파일은 친구 repo 원본 흐름을 유지한다.

기본 흐름:

```text
D435 camera frame 입력
-> YOLO로 유압블럭 bbox/center 검출
-> depth_frame.get_distance(cx, cy)
-> rs2_deproject_pixel_to_point(...)로 camera XYZ 계산
-> OpenCV 표시
```

`--enable-slot-grid`를 켜면 뒤에 다음 단계가 추가된다.

```text
같은 D435 RGB frame에서 박스 bbox 검출
-> 박스 내부 rows/cols slot 분할
-> 유압블럭 center와 slot bbox 매칭
-> occupied / empty / target 상태 계산
-> target slot center 출력
-> JSON/debug image 저장
```

## 실행 명령어

친구 repo 기준 기본 실행:

```powershell
python realsense_xyz_yolo.py --yolo-model best.pt
```

slot grid 활성화:

```powershell
python realsense_xyz_yolo.py --yolo-model best.pt --enable-slot-grid
```

2x3 slot:

```powershell
python realsense_xyz_yolo.py --yolo-model best.pt --enable-slot-grid --rows 2 --cols 3
```

박스 bbox를 수동 지정:

```powershell
python realsense_xyz_yolo.py --yolo-model best.pt --enable-slot-grid --box-bbox 100,80,540,430
```

결과 저장 위치 변경:

```powershell
python realsense_xyz_yolo.py --yolo-model best.pt --enable-slot-grid --output-dir outputs
```

## 저장 결과

기본 저장 파일:

- `outputs/realsense_xyz_yolo_slot_result.json`
- `outputs/realsense_xyz_yolo_slot_debug.jpg`

JSON에는 다음이 포함된다.

- `box.bbox`
- `box.center`
- `slots[].slot_id`
- `slots[].bbox`
- `slots[].center`
- `slots[].status`
- `target_slot.slot_id`
- `target_slot.center`
- `blocks[].bbox`
- `blocks[].center`
- `blocks[].confidence`
- `blocks[].depth_m`
- `blocks[].camera_xyz_m`

## 친구 repo로 옮길 파일

친구 repo에 직접 통합한다면 다음 파일을 옮기면 된다.

필수:

- `realsense_xyz_yolo.py`
- `realsense_block_detector/slot_grid.py`
- `realsense_block_detector/__init__.py`

또는 친구 repo가 flat 구조를 유지해야 한다면 `slot_grid.py`를 repo root로 옮기고, `realsense_xyz_yolo.py`의 import를 다음처럼 바꾸면 된다.

```python
from slot_grid import (
    compute_slot_grid_result,
    parse_bbox,
    save_result_json,
    yolo_results_to_block_detections,
)
```

의존성:

- `pyrealsense2`
- `opencv-python`
- `numpy`
- `ultralytics`

## 기존 로컬 작업물 처리

삭제하지 않고 유지한다.

- `run_grid_box_detection.py`
  - 이미지 테스트 또는 독립 MVP 검증용
  - 최종 기준은 아님

- `realsense_block_detector/slot_grid.py`
  - 최종 통합에 재사용되는 핵심 slot/grid adapter

- `GRID_SLOT_WORK_LOG.md`
  - 초기 MVP 작업 기록
  - 현재 최종 기준은 이 문서가 아니라 `FRIEND_REPO_SLOT_GRID_INTEGRATION.md`

## 다음 단계

1. 실제 D435 화면에서 적재 박스가 잘 보이도록 camera pose를 고정한다.
2. `--enable-slot-grid`로 자동 박스 bbox 검출을 테스트한다.
3. 자동 박스 bbox가 흔들리면 우선 `--box-bbox`로 수동 고정한다.
4. YOLO 유압블럭 bbox center가 slot 안에 들어가는지 확인한다.
5. JSON의 `target_slot.center`를 ROS2 publisher로 넘긴다.
