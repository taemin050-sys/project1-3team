# beamsearch/LTM — 이태민 (Project Manager)

개인 탐색 공간. 구조·도구·방식 자유. `shared`(채점기·가드·SSOT)만 재사용한다.

- 실행 예: `from shared.eval.local_map import ...` · `python -m shared.ssot.build_drug_master ...`
- 공통 데이터: `team/data/raw`(원본)·`team/data/processed`(SSOT) 참조
- 승격 후보 노트북은 의사결정 md 정리해 `team/notebooks/`로.
알약 객체 탐지 모델 (Pill Object Detection with RT-DETR)
이 프로젝트는 주어진 이미지 데이터셋 내의 알약 객체를 탐지하기 위해 Hugging Face의 RT-DETR 모델을 활용하여 파인튜닝(Fine-tuning)을 진행한 파이프라인입니다.   
프로젝트 개요환경: 
Google Colab (GPU 가속 지원)  
데이터셋: 여러 알약 이미지가 포함된 데이터셋(train/val/test) 및 COCO 형식의 JSON 어노테이션 파일.  
주요 모델: PekingU/rtdetr_r50vd_coco_o365 체크포인트를 기반으로 한 RTDetrForObjectDetection 모델.   
주요 기능 및 단계
1. 데이터 준비 및 탐색적 데이터 분석 (EDA)Google Drive를 마운트하고 가상의 Kaggle 경로를 생성하여 데이터를 압축 해제합니다.  OpenCV와 pandas를 이용하여 전체 이미지의 너비, 높이, 비율 통계를 분석합니다.  해상도가 큰 이미지를 분석 모델에 맞게 처리하기 위한 사전 작업을 진행합니다.  
2. 데이터 전처리 및 강력한 증강 기법 (Data Augmentation)
이미지 개수가 상대적으로 적고 크기가 큰 특성을 고려하여 torchvision.transforms.v2를 활용한 다양한 데이터 증강을 적용합니다.  크기 다양성 확보: RandomZoomOut, RandomIoUCrop 기법을 사용하여 작은 물체 인식에 대응합니다.  화질 및 색상 변형: GaussianBlur, RandomAdjustSharpness, ColorJitter 등을 적용하여 다양한 조명 및 초점 환경을 시뮬레이션합니다.  전처리 완료 데이터: 8:2 비율로 분할하여 Train(증강 적용) 및 Validation(증강 미적용) 데이터셋으로 구성합니다.  
3. Custom Dataset 클래스 (PillDatasetDETR / PillDataset)COCO 형식의 통합 JSON 파일에서 어노테이션(bbox, category_id)을 파싱합니다.  BoundingBoxes 객체를 절대 좌표(XYXY) 형식에서 DETR 아키텍처에 맞게 정규화된 좌표(CXCYWH)로 변환합니다.  변환 도중 이미지 바깥으로 나가는 유효하지 않은 바운딩 박스를 필터링하여 안정성을 확보합니다.  
4. 모델 아키텍처 최적화기존 200개였던 num_queries를 100개로 줄여 연산량을 최적화했습니다.  OOM(Out Of Memory) 방지 및 효율적인 학습을 위해 백본 네트워크의 초기 레이어(stage 0, 1 및 embedder) 가중치를 동결(Freeze)하고 디코더 및 상위 레이어만 학습하도록 설정했습니다.  
5. 학습 최적화 및 평가 전략 (Training & Evaluation)Optuna 튜닝: Backbone LR, Head LR, Weight Decay, Unfreeze Stage 등의 하이퍼파라미터를 자동으로 최적화하는 탐색(Search)을 수행합니다.  Mixed Precision: torch.cuda.amp.autocast와 GradScaler를 통해 메모리 절약과 연산 속도 개선을 이끌어냅니다.  성능 평가: torchmetrics를 이용하여 mAP (Mean Average Precision)를 측정합니다.  
최종 학습: Optuna를 통해 찾아낸 Best Parameter를 적용하여 총 80 Epoch의 최종 학습을 진행하며, 최고 성능 달성 시 모델 가중치를 Google Drive에 안전하게 저장합니다.  
