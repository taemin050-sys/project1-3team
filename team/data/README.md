# data/

| 폴더 | 내용 | 커밋 |
| --- | --- | --- |
| `raw/` | Kaggle 원본(이미지·COCO json). `train_images`(232)·`train_annotations`(114 dir)·`test_images`(842) | ❌ 미커밋 |
| `processed/` | SSOT(`drug_master.json`·`class_map.json`)·전처리 산출 | json만 커밋 |

- 다운로드: Kaggle Private Competition → `data/raw/`에 압축 해제.
- Runpod에서는 컨테이너 초기화(Terminate)에 대비해 **재다운로드 또는 Network volume** 사용.
- ⛔ 금지 데이터셋(`TL_2_조합.zip`/`TS_2_조합.zip`) 사용 금지. (docs/01 CR-03)

## 외부 데이터 보강 (docs/07)
- AI Hub 「경구약제 이미지 데이터」의 **단일** 부분 → `data/raw/aihub_single/`에 배치(미커밋).
- 우리 40클래스로 필터: `python -m team.src.data.filter_aihub --aihub-dir team/data/raw/aihub_single --class-map team/data/processed/class_map.json --out team/data/processed/aihub_single_filtered.json`
- 스크립트가 '조합'·`TL_2`·`TS_2` 경로를 자동 차단. 도입 통제: 홍우석(DE).
