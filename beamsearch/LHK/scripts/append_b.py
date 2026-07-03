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
    md("""## 📊 (b) 클래스 분포 · 동시발생

**왜 보는가.** H3(불균형·데이터 천장)와 H6(조합 편향)을 수치로 확인한다. 불균형은 학습 전략(리샘플링·손실 가중·표적 증강)을 정하고, 조합 구조는 리더보드 일반화 리스크를 가늠하는 근거가 된다.

**무엇을 보는가.**
1. **클래스별 객체 수(롱테일)** + 불균형 지표
2. **약품 동시발생** — 어떤 약들이 같은 사진에 함께 담기는지(쌍 빈도·행렬)
3. **조합의 구조성** — 조합이 무작위인지, 같은 조합이 반복되는지""")
)

cells.append(md("### ① 클래스별 객체 수 (롱테일) · 불균형 지표"))

cells.append(
    code("""cid2name = {int(k): (v.get("product_name") or str(k)) for k, v in drug_master.items()}
cls_cnt = ann.category_id.value_counts()                 # category_id -> 객체수 (내림차순)
cls_named = cls_cnt.rename(index=cid2name)
print(f"클래스 {cls_cnt.size}개 | 총 객체 {int(cls_cnt.sum())}개\\n")
print("[상위 8]"); print(cls_named.head(8).to_string())
print("\\n[하위 8]"); print(cls_named.tail(8).to_string())

c = cls_cnt.values.astype(float)
def gini(x):
    x = np.sort(x); n = len(x); cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
print(f"\\n불균형비(max/min) = {c.max()/c.min():.1f}배 | Gini = {gini(c):.3f}")
print(f"top1 점유 = {c.max()/c.sum():.1%} | top5 점유 = {np.sort(c)[::-1][:5].sum()/c.sum():.1%}")
for t in (5, 10, 20):
    print(f"  객체수 < {t} 인 희소 클래스: {(cls_cnt < t).sum()}개")""")
)

cells.append(
    code("""fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
ax[0].bar(range(len(c)), np.sort(c)[::-1], color="#4C78A8")
ax[0].set(title="클래스별 객체 수 (내림차순) — 롱테일", xlabel="클래스 순위", ylabel="객체 수")
ax[0].axhline(c.mean(), ls="--", c="gray", lw=1, label=f"평균 {c.mean():.1f}")
ax[0].legend()
top = cls_named.head(15)[::-1]
ax[1].barh(top.index, top.values, color="#E45756")
ax[1].set(title="상위 15 클래스", xlabel="객체 수")
plt.tight_layout(); plt.show()""")
)

cells.append(md("### ② 약품 동시발생 (쌍 빈도 · 행렬)"))

cells.append(
    code("""from itertools import combinations
sets = ann.groupby("file_name").category_id.apply(lambda s: frozenset(s))

pair = Counter()
for s in sets:
    for a, b in combinations(sorted(s), 2):
        pair[(a, b)] += 1
print(f"동시 등장한 약품 쌍: 서로 다른 {len(pair)}쌍\\n[가장 자주 함께 나온 쌍 top10]")
for (a, b), n in pair.most_common(10):
    print(f"  {n:3d}회  {cid2name[a]}  +  {cid2name[b]}")""")
)

cells.append(
    code("""# 동시발생 행렬 — 가독성 위해 상위 20 빈출 클래스만
top_ids = cls_cnt.head(20).index.tolist()
tset = set(top_ids)
M = pd.DataFrame(0, index=top_ids, columns=top_ids)
for s in sets:
    inter = sorted(x for x in s if x in tset)
    for a, b in combinations(inter, 2):
        M.loc[a, b] += 1; M.loc[b, a] += 1
labels = [cid2name[i][:12] for i in top_ids]
plt.figure(figsize=(9.5, 7.5))
sns.heatmap(M.values, xticklabels=labels, yticklabels=labels, cmap="Reds",
            square=True, linewidths=.3, cbar_kws={"label": "동시 등장 횟수"})
plt.title("약품 동시발생 행렬 (상위 20 클래스)"); plt.tight_layout(); plt.show()""")
)

cells.append(md("### ③ 조합의 구조성 (무작위 vs 반복)"))

cells.append(
    code("""combo_cnt = Counter(sets)                       # 정확히 같은 약품 세트의 빈도
img_per_combo = Counter(combo_cnt.values())
print(f"이미지 {sets.size}장 → 서로 다른 조합(세트) {len(combo_cnt)}개")
print(f"조합당 이미지 수 분포: {dict(sorted(img_per_combo.items()))}")
print(f"조합당 평균 이미지 = {sets.size/len(combo_cnt):.2f}")
# 조합 크기(약품 수) 분포
print("조합 크기(약품 수) 분포:", dict(sorted(Counter(len(s) for s in sets).items())))""")
)

cells.append(md("<<CONCLUSION_B>>"))

nb["cells"].extend(cells)
nbp.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("appended", len(cells), "cells | total", len(nb["cells"]))
