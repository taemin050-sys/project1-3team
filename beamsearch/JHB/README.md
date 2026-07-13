# 💊 JHB

본 폴더는 알약 이미지 데이터셋의 시각화 정제, 합성 데이터(Synthetic) 생성, COCO to YOLO 포맷 변환, 데이터 분포 통계 분석 및 YOLOv11 모델 학습까지의 모든 파이프라인을 독립적으로 수행할 수 있는 유틸리티 스크립트 모음입니다.

팀 아키텍처 규칙을 준수하여 대용량 데이터 원본과 합성 결과물은 `team/data/raw/` 및 `team/outputs/`를 바라보도록 상대경로(`pathlib`) 처리가 완료되어 있습니다.

---

## 📂 스크립트 기능 소개 (Features)

### 1. 데이터 정제 및 파이프라인 통합 (Data Refinement & Pipeline)
* **`check_classes.py`** (구 `visualize_by_class`)
  * 원본 및 합성 데이터셋을 클래스(알약 제품명)별로 정렬하여 16분할 그리드로 시각화합니다. 마우스 좌클릭으로 불량 데이터를 선택해 즉시 영구 삭제 정제할 수 있습니다.
* **`yolo11s.py`** (구 `main` 파이프라인 스크립트)
  * 순정 알약 데이터셋을 정밀하게 8:2(Train/Val)로 분할하고, 제작된 합성 데이터 3,000장을 오직 Train 세트에만 100% 융합 이식하여 최종 YOLO 학습용 컨테이너를 구성합니다.

### 2. 합성 데이터 생성 및 변환 (Synthetic Dataset Tools)
* **`build_synthetic_dataset.py`**
  * 베이스 배경 이미지 위에 크롭된 알약 객체들을 무작위 배치, 회전, 변형하여 대량의 오그멘테이션 합성 이미지를 생성하고 통합 COCO JSON 라벨을 출력합니다.
* **`convert_synthetic_coco_to_yolo.py`**
  * 합성 엔진이 생성한 통합 `synthetic_annotations.json` 파싱하여 YOLO 모델이 즉시 학습할 수 있는 낱개 `.txt` 라벨 파일들로 역정규화 변환합니다.

### 3. 데이터 전수조사 및 검증 (Data Analysis & QA)
* **`count_classes.py`**
  * YOLO 포맷의 `.txt` 라벨들을 전수조사하여 93개 클래스별 알약 개수, 비율(%)을 산출하고 특정 클래스 쏠림 현상을 감지하는 위험 경고 시스템을 가동합니다.
* **`qa_visualize.py`**
  * 합성된 결과물 이미지 위에 COCO JSON 기반 바운딩 박스(BBox)와 알약 이름을 반투명하게 오버레이 렌더링하여 품질(그림자, 겹침 등)을 시각적으로 검증합니다.
* **`show_qa.py`**
  * `qa_visualize.py`가 생성한 검증용 이미지들을 슬라이드쇼 형태로 화면에 띄웁니다. `스페이스바`로 다음 사진을 넘기며 알약 데이터 상태를 빠르게 모니터링할 수 있습니다.

### 4. 수량 체크
* **`check_train.py` / `check_val.py`**
  * 최종 분할 및 융합 작업이 완료된 Train 및 Val 컨테이너 폴더를 실시간으로 스캔하여 타겟팅이 제대로 되었는지 경로 상태를 체크합니다.
* **`count_images.py` / `count_txt.py`**
  * 특정 디렉토리 내의 순수 이미지 파일(`.png`, `.jpg`) 개수와 라벨 파일(`.txt`) 개수를 독립적으로 빠르게 집계하여 파일 쌍이 맞지 않는 유실 데이터가 있는지 검증합니다.

---

## 🚀 권장 가동 순서 (Pipeline Workflow)

데이터셋을 처음부터 정제하고 모델 학습까지 달리는 추천 워크플로우입니다.

1. **데이터 정제 및 상태 파악:** `check_classes.py` ➡️ `count_classes.py`
2. **합성 데이터셋 빌드 및 변환:** `build_synthetic_dataset.py` ➡️ `convert_synthetic_coco_to_yolo.py`
3. **합성 품질 시각화 QA 검증:** `qa_visualize.py` ➡️ `show_qa.py`
4. **최종 데이터 셋 8:2 분할 및 합성 데이터 융합:** `yolo11s.py`
5. **최종 수량 교차 검증:** `count_images.py` / `count_txt.py`