# 데이터 출처 · 보강 — 문서 07

> **코드잇 스프린트 AI 엔지니어링 12기 · 초급 프로젝트 · 3팀**
> 목적: 저데이터(train 232장) 해소를 위한 외부 데이터 조사·도입 기준. (실행전략 ③①, 02 PRD E5)
> 조사: 이형기 · 데이터 도입 통제: 홍우석(DE) · 작성일 2026-06-29

---

## 0. 결론 (먼저)

- **1차 보강 = AI Hub 「경구약제 이미지 데이터」의 '단일(single)' 부분.** 우리 대회 원본과 **같은 출처·같은 메타 스키마**(`dl_idx`/`dl_name`)라 `category_id` 매핑이 자동 정합된다. → 우리 40개 클래스만 골라 무손실 합류 가능.
- **글로벌 공개 데이터(Roboflow/Kaggle) = 2차·보조.** 대부분 단일 "pill" 클래스로 통합돼 우리 40개 클래스에 매핑 불가 → 클래스별 보강엔 부적합.
- **금지 경계 엄수:** 금지 2종은 '조합' 세트의 `TL_2_조합.zip`/`TS_2_조합.zip`. '단일' 세트는 별개이며 사용 가능.

---

## 1. 금지 데이터셋 경계

| 구분 | 사용 | 이유 |
| --- | --- | --- |
| 경구약제 **단일** 데이터 | ✅ 권장 | 금지 파일과 별개, 스키마 호환 |
| 경구약제 **조합** — `TL_2_조합.zip`/`TS_2_조합.zip` | ⛔ 절대 금지 | 대회 train/test 원본(추적·적발) |
| 경구약제 **조합** — 그 외 분할 | ⚠️ 회피 권장 | test가 조합 세트에서 파생 → 중복 위험 |

> **예외(주강사 확인):** AI Hub **조합 validation** 데이터는 **오토라벨링 검증용으로만** 사용 가능(학습 직접 사용 X). 오토라벨 결과(bbox)를 정답과 대조하는 용도에 한함. — 주강사 확인 완료.

> 다운로드 페이지의 **파일 목록**에서 zip 이름을 직접 확인해 금지 2종을 배제한다. 코드 차원에서도 경로 가드(`team/src/data/filter_aihub.py`)로 '조합'·`TL_2`·`TS_2`를 차단.

---

## 2. 1차 — AI Hub 「경구약제 이미지 데이터」(단일)

- **출처:** AI Hub, `경구약제 이미지 데이터` (dataSetSn=576) — 우리 대회 데이터의 원본
- **링크:** https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=576
- **규모·라벨:** 전문의약품 3,143종 + 일반의약품 1,857종(총 5,000종). 다양한 각도·조명·배경. 라벨 = **바운딩 박스 + 약제/촬영 메타정보**. 단일 이미지와 3~4개 조합 이미지를 모두 구축.

### 왜 최적인가 — 스키마 호환
우리 `drug_master`를 만든 그 메타 스키마(`dl_idx`, `dl_name`, `di_etc_otc_code` 등)를 그대로 사용 → **`dl_idx`로 우리 40개 클래스에 정확히 매칭**되는 단일 알약 이미지만 선별 가능. (글로벌 데이터셋이 줄 수 없는 강점)

### 활용 전략
1. 단일 이미지를 **우리 40 클래스(`dl_idx`)로 필터링** → 클래스별 학습 샘플 증강 (저데이터 직접 해소)
2. 단일 → **멀티 합성**: **구조적 Copy-Paste 증강**(`team/src/data/augmentor.py`)으로 "이미지당 최대 4개" 분포 인공 생성 — 흰배경 투명화 → 그리드 분할(오클루전 차단) → 알약별 기하증강 → 알파합성 + bbox 전역좌표. 저데이터 핵심 해결책.
3. 같은 `dl_idx` → `category_id` 자동 정합 → SSOT(문서 04)에 무손실 합류

### 다운로드 방법
- AI Hub 회원가입·로그인 → 데이터셋 다운로드 승인 → **aihubshell**(스크립트)로 리눅스/WSL/Mac에서 조회·다운로드.
- 파일은 **분할 압축** → 병합 필요: `find "<경로>" -name "*.zip.part*" -print0 | sort -zt'.' -k2V | xargs -0 cat > "<파일>.zip"`
- 받은 원본은 `data/raw/aihub_single/`에 배치(미커밋). Runpod에서는 재다운로드 또는 Network volume.
- ⚠️ 접근 유형(일반 다운로드 / 보건의료 안심존) 다운로드 페이지에서 확인.

---

## 3. 2차 — 글로벌 공개 데이터셋 (보조)

Roboflow Universe·Kaggle에 알약 검출 데이터가 다수 있고 COCO/YOLO 포맷으로 바로 받을 수 있음(예: Roboflow "Pill Detection", "Medicine" 등).

**한계(솔직히):** 대부분 전 클래스를 하나의 "pill"로 통합하거나 다른 약을 다룸 → **우리 40개 한국 약 클래스에 매핑 불가**. 클래스별 보강엔 부적합하고, 일반 localization 감각은 COCO 전이학습이 이미 제공. → **1차 소진 후 여력 있을 때만 보조 고려.**

---

## 4. 도입 전 검증 체크리스트

1. **금지 2종 배제** — 파일명 단위로 `TL_2_조합`/`TS_2_조합` 미포함 확인 (다운로드 전·후)
2. **조합 세트 회피** — 부득이 사용 시 test 이미지와 해시 중복 검증 필수
3. **클래스 매핑** — `dl_idx → category_id`로 우리 40클래스만 필터 (문서 04 SSOT)
4. **분포 점검** — 병합 후 클래스 분포·이미지 통계 확인(편향 리스크) — 홍우석 단일 통제
5. **출처 고지** — AI Hub 데이터는 활용 시 출처 명시 의무 → 보고서·README에 명기

---

## 5. 필터 스크립트

`team/src/data/filter_aihub.py` — 다운로드한 AI Hub '단일' 데이터를 우리 40클래스로 필터링하고 금지 경로를 차단. 사용법은 스크립트 상단 docstring 참조.

```bash
python -m team.src.data.filter_aihub \
    --aihub-dir data/raw/aihub_single \
    --class-map data/processed/class_map.json \
    --out data/processed/aihub_single_filtered.json \
    --test-images data/raw/test_images   # 중복 방지(선택)
```

> 의존: `class_map.json`(문서 04 §6, `build_drug_master`로 생성)이 먼저 있어야 함.

---

## 6. 다음 액션

1. (홍우석) AI Hub 「경구약제 이미지 데이터」 **단일** 다운로드 → `data/raw/aihub_single/`
2. `build_drug_master` → `class_map.json` 생성 → `filter_aihub`로 40클래스 필터
3. 분포 점검 → SSOT 합류 → mosaic 합성 실험(E5 ablation, 문서 02)
