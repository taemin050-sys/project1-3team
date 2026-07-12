# LHK 구현 소스 — 재현 가이드 (경구약제 검출)

> 기획 → 대회 → 상용화 전 과정의 **실행 파일(.py)** 정리본. 정본은 멀티머신 툴킷 `github.com/exobiz7/lab-kit`
> (`projects/healtheat/scripts`). 여기(`src/`)는 팀 검토·재현용 스냅샷.
> 무거운 데이터·가중치·원본은 **커밋하지 않음** → `../reports/TEAM_DRIVE_ASSETS.md`(구글드라이브 링크/리스트) 참조.

## 실행 환경
- Python 3.11 (conda `codeit`). 주요 의존: `ultralytics`, `pycocotools`, `ensemble-boxes`, `opencv-python`, `pillow`, `pandas`, `matplotlib`.
- 디바이스: Apple Silicon **MPS**(맥북·맥스튜디오) / **CUDA**(Colab, RT-DETR). 공통코드 `cuda>mps>cpu`.
- 경로 어댑터: `paths.py`(+`labkit.py`·`config.json`)가 머신별 데이터 위치를 자동 해석. 재현 시 `config.json`의 `roots`에 본인 데이터 경로 추가.
- 실행 래퍼: `run.sh`(절전방지 caffeinate + 로그). 예) `bash run.sh kfold_exp.py <이름>`.

## 파이프라인 단계별 스크립트

| 단계 | 스크립트 | 역할 |
| --- | --- | --- |
| **EDA·베이스라인** | `e1_baseline.py`·`e1_smoke_train.py`·`e1_post.py` | E1 베이스라인 학습·평가(COCO→YOLO→mAP 하니스) |
| | `tv_baselines.py`·`ultra_baselines.py`·`fasterrcnn_fix.py` | 다중모델 공정비교(RetinaNet/FCOS/RT-DETR/FasterRCNN, MPS 발산 fix) |
| **증강·합성** | `augment.py`·`smoke_augment.py` | 온라인 증강 프로파일(default/strong/realistic) |
| | `e2_synth_ablation.py`·`e3_synth2500.py`·`e4_1500_696style.py` | 합성 물량·분포 애블레이션(플래토 검증) |
| | `build_realcopy_manifest.py`·`gen_realcopy_manifests_perfold.py`·`gen_clean_backgrounds.py` | real copy-paste 매니페스트(누수안전 per-fold)·클린 배경 |
| **추출·전처리** | `aihub_extract_cover.py` | ★AI Hub 조합 전량추출(10,489장/116클래스, 커버리지) |
| | `aihub_extract_inpaint.py`·`aihub_extract_full.py`·`aihub_extract_poc.py` | 인페인트 마스킹 추출(비-우리 알약 seamless 제거) |
| | `prep_yolo.py`·`gen_gt_corrections.py` | COCO→YOLO 변환·GT 누락 8건 보정 |
| **평가(K-fold)** | `kfold_exp.py` | ★파라미터화 실험러너(EXP_MODEL/SYNTH/EPOCHS/EXCLUDE/CLASSMAP) |
| | `kfold_sweep.py`·`kfold_resolution.py`·`test_time_tune.py` | 멀티폴드 스윕·해상도·test-time 튜닝 |
| **라벨정리** | `label_audit.py` | ★2트랙 자동 라벨감사(모델 불일치+기하) |
| | `error_analysis.py` | 오류분석(혼동·클래스별 AP·오류 컨택트시트) |
| **최종 학습·제출** | `final_submission.py` | ★전량 학습→test842 예측→제출 CSV(EXP_* env로 모델·데이터 선택) |
| **앙상블** | `wbf_ensemble.py` | ★WBF 앙상블(submit/eval/foldpred) |
| **RT-DETR** | `build_rtdetr_final_bundle.py`·`build_rtdetr_nb_cover.py`·`build_rtdetr_nb_final.py` | Colab 번들·재개형 노트북 생성 |
| **시각화** | `make_class_catalog.py`·`make_cover_catalog.py`·`make_audit_diagram.py`·`make_scrum_grid*.py` | 클래스 카탈로그·감사 다이어그램·의심 그리드 |
| **인프라** | `paths.py`·`labkit.py`·`config.json`·`run.sh` | 멀티머신 경로 어댑터·git-버스·실행 래퍼 |

## 재현 순서 (요약)
```bash
# 0) 데이터 준비: AI Hub 조합/단일 + 대회 기본데이터 → config.json roots에 경로 등록 (원본은 Drive 리스트 참조)
# 1) 추출: python aihub_extract_cover.py            # cover116 생성
# 2) 클래스맵: (SSOT) class_map_cover.json 확인
# 3) 라벨감사: AUDIT_BEST=<best.pt> AUDIT_SYNTH=kaggle_aihub_cover116 python label_audit.py → exclude 목록
# 4) 최종학습: EXP_CLASSMAP=class_map_cover.json EXP_SYNTH=...cover116 EXP_MODEL=yolo11m.pt \
#             EXP_EXCLUDE=exclude_cover_all.txt bash run.sh final_submission.py final_11m_cover118_clean
# 5) 앙상블: WBF_MODE=submit WBF_INPUTS="11m.csv:3,rtdetr.csv:1" WBF_SKIP=0.3 python wbf_ensemble.py
```

## 결과 파일
- `results/*.jsonl` — 머신별 k-fold 실험 기록(폴드평균 mAP 등, 리포트 표의 원천).
- 최종 제출 CSV·class_map: `../reports/data/`.
- **모델 가중치·체크포인트·대용량 데이터셋·번들 zip → `../reports/TEAM_DRIVE_ASSETS.md`**(무거워 미커밋, Drive 공유).
