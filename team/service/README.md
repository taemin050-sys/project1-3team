# service/

웹 서비스. 서빙 트랙 단일 경량 모델 + `drug_master` 조인 → 약 정보 응답. (docs/05) · 주담당: 최중열

- **T1 (MVP):** Gradio 단일 앱 — 업로드 → 박스 오버레이 + 약 정보 카드
- **T2 (심화):** FastAPI 백엔드 + 프론트 + 로깅(DB)
- `serving_bundle/`: 모델 가중치 + `class_map.json` + `drug_master.json` 동봉 → 버전 불일치 차단. (docs/05 §2)
  - 가중치(`*.pt`)는 미커밋. 번들 구성·배포 절차는 docs/05 참조.
- `app.py`: T1 Gradio 앱(구현) — 업로드 → 박스 오버레이 + **약 정보 카드**(drug_master 조인). `python service/app.py`로 실행(번들 필요).
