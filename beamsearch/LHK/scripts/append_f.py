import json
from pathlib import Path

nbp = Path(
    "/Users/macbook/dev/learning/codeit/01_Proj_HealthEat_cla/project1-3team/beamsearch/LHK/01_eda_domain_anchor.ipynb"
)
nb = json.loads(nbp.read_text(encoding="utf-8"))


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": s,
    }


cells = []
cells.append(
    md("""## 🧭 (f) EDA 종합 · 가설 갱신

(b)(c)(e)에서 확인한 사실을 **가설 검증 → 데이터 계약·실행 설계**로 묶는다. 이 절이 베이스라인(E1)과 이후 실험의 출발 규격이 된다.""")
)

cells.append(
    md("""### 가설 검증표 (H1~H6)

| 가설 | 결과 | 핵심 근거 | 함의 |
| --- | --- | --- | --- |
| **H1 저데이터** | ✅ 확정 | train 232장·763객체, 56클래스 중 48개가 객체<20 | 전이학습·증강·(2) 물량 보강 필수 |
| **H2 고-IoU→localization** | ✅ 확정(전제) | 박스 상대면적 중앙값 4%, 640서도 한 변 ≈116px(검출 충분) | 해상도 가치 = "작은물체"가 아니라 **엣지 정밀** → 1024·CIoU/DFL |
| **H3 불균형·천장** | ✅ 확정 | Gini 0.497, 불균형 51배, 객체<5 클래스 17개 | 리샘플·손실가중, **희소 17클래스=천장 후보**(과투자 금지) |
| **H4 색 불안정** | ⚠️ **부분 반전** | (2)는 WB 불안정하나 **(1)은 조명 주백색 단일→색 안정** | **대회 도메인선 색 사용 가능**, (2) 혼합 시 WB 정규화 전제 |
| **H5 (1)↔(2) 도메인갭** | ✅ 확정(방향 반대) | (1) 배경/조명 단일 ↔ (2) 배경·WB 광범위 | (2)를 **(1) 도메인으로 정규화**해 사용, 원색 배경 학습 지양 |
| **H6 조합 편향** | ✅ 확정 | 허브약 `003351`이 66% 등장, 조합 반복(232→122) | **조합 단위 Group K-Fold**(누수 방지), 허브약 편향 주의 |""")
)

cells.append(
    md(
        "### 검증: 조합 단위 Group K-Fold (누수·희소 클래스 확인)\n\n같은 조합(약품 세트)을 각도만 바꿔 반복 촬영했으므로, 단순 무작위 분할은 **train/val 누수**를 만든다. 조합을 그룹으로 묶어 분할하고, 각 fold의 클래스 커버리지를 확인한다."
    )
)
cells.append(
    code("""# 조합(frozenset) = 그룹. 의존성 없이 수동 Group K-Fold (sklearn은 학습 단계에서 도입)
img_names = sets.index.tolist()
combo_of = {c: i for i, c in enumerate(dict.fromkeys(sets.values))}
groups = np.array([combo_of[sets[fn]] for fn in img_names])
img_cls = {fn: set(g.category_id) for fn, g in ann.groupby("file_name")}

rng = np.random.default_rng(SEED)
uniq = np.array(sorted(set(groups))); rng.shuffle(uniq)
folds = np.array_split(uniq, 5)
g2fold = {g: k for k, fl in enumerate(folds) for g in fl}
img_fold = np.array([g2fold[g] for g in groups])

print(f"이미지 {len(img_names)}장 · 조합(그룹) {len(uniq)}개 → 5-fold\\n")
for k in range(5):
    va = [i for i in range(len(img_names)) if img_fold[i] == k]
    tr = len(img_names) - len(va)
    va_cls = set().union(*[img_cls[img_names[i]] for i in va]) if va else set()
    print(f"fold{k}: train_img={tr:3d}  val_img={len(va):3d}  | val 클래스 {len(va_cls):2d}/56 (val 부재 {56-len(va_cls)}개)")
print("\\n→ 같은 조합이 train/val로 쪼개지지 않음(누수 방지). 단 fold마다 희소 클래스가 val에서 빠져 per-class AP는 불안정.")""")
)

cells.append(md("<<CONCLUSION_F>>"))

nb["cells"].extend(cells)
nbp.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("appended", len(cells), "cells | total", len(nb["cells"]))
