# 팀 구글드라이브 업로드 대상 — 무거운 자산 (LHK)

> 리포트는 이 자산들을 **경로/링크로 참조**한다(git 미커밋). 아래 정확 경로로 KAI가 팀 드라이브에 업로드.
> 경량 자산(리포트 이미지·소 CSV·노트북)은 `beamsearch/LHK/`에 커밋됨.

## 1. 최종 제출 CSV (캐글 [AI12])
- `…/multi-machine/projects/healtheat/runs/final/submission_wbf_11m_rtdetr_skip03.csv` — **WBF 앙상블 0.9994(best)**
- `…/submission_wbf_11m_rtdetr_skip02.csv` — 앙상블 0.9994(동률)
- `…/submission_yolo11m_full232_synth696_aihubfull_cover118_clean.csv` — 11m clean 0.9988
- `…/submission_rtdetr_cover_clean.csv` — RT-DETR solo 0.9935

## 2. 모델 가중치 (best.pt)
- 11m cover118 clean: `…/runs/final/yolo11m_full232_synth696_aihubfull_cover118_clean/weights/best.pt` (~40MB, Mac Studio 로컬)
- 11s cover118 clean: `…/runs/final/yolo11s_full232_synth696_aihubfull_cover118_clean/weights/best.pt`
- RT-DETR cover clean: `…/rtdetr_cover_run/rtdetr_cover/weights/best.pt` (Drive)

## 3. 커버리지 데이터셋
- `…/01_data/processed/kaggle_aihub_cover116/` (10,489장, 116클래스, ~2.7GB) — 마운트 `/Volumes/SSD 4T/…/processed/kaggle_aihub_cover116`
- `class_map_cover.json` / `aihub_drug_names.json` (경량, LHK/reports/data에도 사본)

## 4. RT-DETR 번들
- `…/projects/healtheat/rtdetr_cover_clean_bundle.zip` (~2.3GB)

## 5. 풀해상도 시각자산 (리포트는 다운스케일본 커밋)
- `…/projects/healtheat/label_audit/scrum_suspects_grid.png` (11MB) · `scrum_suspects_grid_original.png` · `class_catalog_cover118.png`

## 6. Codex 산출물 (참조, 원본 보존)
- MFDS 수집 이미지 25,326장: `/Volumes/SSD 4T/mfds-pill-image-collector/data/images/`
- 프리즈 벤치마크·도메인랜덤 검출기 데이터: Codex autolabel `outputs/`(Mac Studio 로컬)
- Codex 레포: `github.com/exobiz7/{health-eat-pill-autolabeling, mfds-pill-image-collector}` + `aihub-image-preprocessing`(로컬)
