# beamsearch/HWS — 홍우석 (Data Engineer)

개인 탐색 공간. 구조·도구·방식 자유. `shared`(채점기·가드·SSOT)만 재사용한다.

- 실행 예: `from shared.eval.local_map import ...` · `python -m shared.ssot.build_drug_master ...`
- 공통 데이터: `team/data/raw`(원본)·`team/data/processed`(SSOT) 참조
- 승격 후보 노트북은 의사결정 md 정리해 `team/notebooks/`로.

## 알약 객체 탐지 실험

동일한 leakage-safe Kaggle + AIHub 데이터로 YOLO11s와 RT-DETR-L을 학습하고 Kaggle 제출 결과를 비교했다.

| 모델 | 입력 크기 | Kaggle 점수 | 실행 코드 | 설명 |
| --- | ---: | ---: | --- | --- |
| YOLO11s | 1280 | **0.987** | [Kaggle 노트북](experiment/yolo/yolo11s_kaggle.ipynb) | [README](experiment/yolo/README.md) |
| RT-DETR-L | 960 | **0.994** | [Kaggle 노트북](experiment/RT-DETR/rtdetr_l_kaggle.ipynb) | [README](experiment/RT-DETR/README.md) |

confidence별 제출 통계와 모델 간 차이는 [결과 비교 문서](experiment/results/README.md)에 정리했다.

## 실험 구조

```text
experiment/
├── yolo/
│   ├── README.md
│   └── yolo11s_kaggle.ipynb
├── RT-DETR/
│   ├── README.md
│   └── rtdetr_l_kaggle.ipynb
└── results/
    └── README.md
```

## 아티팩트 보관

모델 가중치(`*.pt`), 제출 CSV, 데이터셋, 학습 로그는 Git에 포함하지 않는다. 최종 가중치와 제출 파일은 Kaggle Dataset 또는 팀 공유 스토리지에 별도로 보관하고, 재현에 필요한 설정과 파일명은 각 실험 README에 기록한다.
