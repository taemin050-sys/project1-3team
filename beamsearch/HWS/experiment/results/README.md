# YOLO11s와 RT-DETR-L 결과 비교

Kaggle 알약 객체 탐지 과제에서 동일한 데이터 구성을 사용해 학습한 YOLO11s와 RT-DETR-L의 제출 결과를 정리합니다.

## 실험 조건

- 학습 이미지: 5,824장
- 검증 이미지: 45장
- 클래스: 93개
- 테스트 이미지: 842장
- 학습 데이터: leakage-safe Kaggle + AIHub 조합 데이터
- 제출 최대 검출 수: 이미지당 10개

두 모델은 같은 COCO category 목록을 기준으로 학습했습니다. 학습용 class index는 `0`부터 연속적으로 구성하고, 제출 시 원래 COCO `category_id`로 복원했습니다.

## Kaggle 결과

| 모델 | 입력 크기 | Kaggle 점수 |
| --- | ---: | ---: |
| YOLO11s | 1280 | **0.987** |
| RT-DETR-L | 960 | **0.994** |

RT-DETR-L이 YOLO11s보다 `0.007` 높은 점수를 기록했습니다.

최종 검증 mAP는 학습 결과 파일이 현재 저장소에 포함되지 않아 이 문서에서 비교하지 않습니다. 이 표에는 확인된 Kaggle 제출 점수만 기록했습니다.

## 제출 후보 통계

### YOLO11s

| Confidence | 행 수 | 검출 이미지 | 예측 클래스 | 이미지당 평균 박스 | 평균 score |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 3,267 | 842 | 77 | 3.880 | 0.9280 |
| 0.15 | 3,259 | 842 | 77 | 3.871 | 0.9300 |
| 0.25 | 3,252 | 842 | 77 | 3.862 | 0.9315 |

YOLO11s는 confidence를 `0.10`에서 `0.25`로 높여도 예측 수가 크게 변하지 않았습니다. 대부분의 예측 score가 높은 구간에 모여 있음을 보여줍니다.

### RT-DETR-L

| Confidence | 행 수 | 검출 이미지 | 예측 클래스 | 이미지당 평균 박스 | 평균 score |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 8,420 | 842 | 91 | 10.000 | 0.4597 |
| 0.15 | 5,473 | 842 | 88 | 6.500 | 0.6440 |
| 0.25 | 3,628 | 842 | 78 | 4.309 | 0.8795 |

RT-DETR-L은 confidence 설정에 따라 제출 행 수가 크게 달라졌습니다. `0.05` 후보는 모든 이미지에서 `max_det=10`에 도달하므로 오탐이 포함될 가능성이 높고, threshold 선택의 영향이 YOLO11s보다 큽니다.

## 해석

- 두 모델 모두 842장의 테스트 이미지에서 최소 한 개 이상의 객체를 검출했습니다.
- YOLO11s는 confidence 후보 간 예측 개수가 거의 같아 높은 score 예측 위주로 결과가 형성됐습니다.
- RT-DETR-L은 낮은 confidence 구간에 추가 후보가 많았으며, 적절한 threshold에서는 YOLO11s보다 더 많은 검출을 유지했습니다.
- RT-DETR-L의 높은 Kaggle 점수는 추가 검출 중 유효한 객체가 포함되어 recall 측면에서 이점을 얻었을 가능성이 있습니다. 이는 제출 CSV 통계를 바탕으로 한 추정입니다.
- Kaggle 점수와 제출 CSV만으로 과적합 여부를 확정할 수 없습니다. 실제 촬영 이미지에서는 조명, 반사, 배경, 가림과 같은 도메인 차이를 별도로 평가해야 합니다.

## 재현 코드

- [YOLO11s Kaggle 노트북](../yolo/yolo11s_kaggle.ipynb)
- [YOLO11s 실행 설명](../yolo/README.md)
- [RT-DETR-L Kaggle 노트북](../RT-DETR/rtdetr_l_kaggle.ipynb)
- RT-DETR-L 실행 설명은 `../RT-DETR/README.md`에 정리할 예정입니다.

## 산출물 보관

다음 파일은 Git에 포함하지 않고 Kaggle Dataset 또는 팀 공유 스토리지에 보관합니다.

- `yolo11s_1280_best.pt`
- `rtdetr_l_960_best.pt`
- `submission_yolo11s_1280_conf100_max10.csv`
- `submission_yolo11s_1280_conf150_max10.csv`
- `submission_yolo11s_1280_conf250_max10.csv`
- `submission_rtdetr_l_960_conf050_max10.csv`
- `submission_rtdetr_l_960_conf150_max10.csv`
- `submission_rtdetr_l_960_conf250_max10.csv`

## 제출 형식

```text
annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
```

`image_id`는 이미지 파일명의 숫자를 사용하고, `annotation_id`는 전체 행에 대해 중복되지 않도록 순서대로 생성했습니다.
