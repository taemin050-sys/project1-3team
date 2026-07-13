# YOLO11s 알약 객체 탐지

Kaggle 알약 객체 탐지 데이터에 YOLO11s를 학습하고, 대회 제출 형식의 CSV를 생성하는 실험입니다.

## 파일

- `yolo11s_kaggle.ipynb`: 데이터 준비부터 학습, 추론, 제출 파일 검증까지 실행하는 Kaggle 노트북

## 데이터

- 학습 이미지: 5,824장
- 검증 이미지: 45장
- 클래스: 93개
- 테스트 이미지: 842장
- 학습 데이터: leakage-safe Kaggle + AIHub 조합 데이터

원본 어노테이션은 COCO 형식입니다. 노트북에서 Ultralytics 학습 형식으로 변환하며, 비연속적인 COCO `category_id`를 `0`부터 시작하는 class index로 매핑합니다. 제출 파일을 만들 때는 `class_map.json`을 사용해 원래 `category_id`로 복원합니다.

## 실행 흐름

1. Kaggle 환경과 GPU를 확인합니다.
2. 학습 데이터와 대회 테스트 이미지 경로를 자동 탐색합니다.
3. COCO 어노테이션을 Ultralytics 라벨 형식으로 변환합니다.
4. 이미지와 라벨 쌍, 클래스 범위, 정규화 좌표를 검증합니다.
5. 사전학습된 YOLO11s를 학습합니다.
6. `best.pt`로 테스트 이미지에 스트리밍 추론을 수행합니다.
7. confidence별 제출 후보 CSV를 생성하고 제출 컬럼을 검증합니다.

## 주요 설정

| 항목 | 값 |
| --- | ---: |
| 모델 | YOLO11s |
| 이미지 크기 | 1280 |
| 배치 크기 | 16 |
| 옵티마이저 | AdamW |
| 초기 학습률 | 0.003971084710792475 |
| Weight decay | 1.3783237455007187e-05 |
| IoU threshold | 0.70 |
| 최대 검출 수 | 이미지당 10개 |
| Seed | 42 |

제출 후보 confidence는 `0.10`, `0.15`, `0.25`로 설정했습니다.

## 결과

| 모델 | 입력 크기 | Kaggle 점수 |
| --- | ---: | ---: |
| YOLO11s | 1280 | **0.987** |

## 산출물

노트북 실행 결과는 `/kaggle/working`에 생성됩니다.

- `yolo11s_1280_best.pt`: 검증 성능이 가장 좋은 가중치
- `submission_yolo11s_1280_conf100_max10.csv`
- `submission_yolo11s_1280_conf150_max10.csv`
- `submission_yolo11s_1280_conf250_max10.csv`
- `submission.csv`: 기본 제출 후보

가중치, 제출 CSV, 학습 로그는 용량과 실행 환경 의존성 때문에 Git에 포함하지 않습니다. 최종 가중치는 Kaggle Dataset 또는 팀 공유 스토리지에 별도로 보관합니다.

## 제출 형식

```text
annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
```

- `image_id`는 테스트 이미지 파일명의 숫자를 사용합니다.
- `annotation_id`는 전체 행에 대해 중복되지 않는 순번을 사용합니다.
- `bbox`는 픽셀 단위의 `x`, `y`, `width`, `height` 형식입니다.
