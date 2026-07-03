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
    md("""## 📐 (c) bbox 크기 · 위치 · 개수

**왜 보는가.** 고-IoU 지표(H2)에서는 박스 크기·위치 분포가 곧 설계 근거다. 알약이 화면에서 얼마나 크고 어디에 놓이는지가 **입력 해상도(E3: 640→1024)·스케일·후처리**를 좌우한다. 더불어 (b)에서 찾은 **품질플래그 8장**(파일명 약품 수 > 실제 bbox 수)을 실제 이미지로 확인한다.

**무엇을 보는가.** ① bbox 크기(면적·종횡비)·이미지 대비 상대 크기, ② 중심 위치 분포(중앙 편중 여부), ③ 이미지당 객체 수 + 품질플래그 8장 육안 확인.""")
)

cells.append(md("### ① bbox 크기 · 종횡비 · 상대 크기"))
cells.append(
    code("""W, H = 976, 1280
arr = np.array(ann.bbox.tolist(), dtype=float)
b = pd.DataFrame(arr, columns=["x", "y", "w", "h"])
b["area"] = b.w * b.h
b["rel_area"] = b.area / (W * H)
b["aspect"] = b.w / b.h
b["side_px"] = np.sqrt(b.area)

def pct(s): return {q: round(float(s.quantile(q)), 2) for q in (0, .25, .5, .75, 1.0)}
print("bbox w(px):", pct(b.w))
print("bbox h(px):", pct(b.h))
print("종횡비 w/h :", pct(b.aspect))
print("상대 면적(% of image):", {k: f"{v*100:.1f}%" for k, v in pct(b.rel_area).items()})
print(f"박스 한 변 근사 side_px 중앙값 {b.side_px.median():.0f}px "
      f"(이미지 짧은변 {W}px 대비 {b.side_px.median()/W:.0%})")
print(f"입력 리사이즈 시 중앙값 박스 한 변 ≈ 640: {b.side_px.median()*640/H:.0f}px / 1024: {b.side_px.median()*1024/H:.0f}px")""")
)

cells.append(
    code("""fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(b.rel_area * 100, bins=30, color="#4C78A8"); ax[0].set(title="bbox 상대 면적 (% of image)", xlabel="%", ylabel="객체 수")
ax[1].hist(b.aspect, bins=30, color="#72B7B2"); ax[1].axvline(1, ls="--", c="gray", label="정사각(w=h)"); ax[1].legend(); ax[1].set(title="종횡비 (w/h)", xlabel="w/h")
ax[2].hist(b.side_px, bins=30, color="#54A24B"); ax[2].set(title="박스 한 변 근사 (px)", xlabel="px")
plt.tight_layout(); plt.show()""")
)

cells.append(md("### ② 중심 위치 분포 (중앙 편중 여부)"))
cells.append(
    code("""cx = (b.x + b.w / 2) / W
cy = (b.y + b.h / 2) / H
plt.figure(figsize=(5.6, 6.6))
plt.hist2d(cx, cy, bins=28, range=[[0, 1], [0, 1]], cmap="magma")
plt.gca().invert_yaxis(); plt.colorbar(label="객체 수")
plt.title("bbox 중심 위치 분포 (정규화)"); plt.xlabel("x / W"); plt.ylabel("y / H")
plt.tight_layout(); plt.show()
print(f"중심 x 중앙값 {cx.median():.2f} (IQR {cx.quantile(.25):.2f}~{cx.quantile(.75):.2f})")
print(f"중심 y 중앙값 {cy.median():.2f} (IQR {cy.quantile(.25):.2f}~{cy.quantile(.75):.2f})")""")
)

cells.append(md("### ③ 이미지당 객체 수 + 품질플래그 8장 육안 확인"))
cells.append(
    code("""cnt = ann.groupby("file_name").size()
def ncodes(fn): return len([c for c in fn.split("_")[0].split("-")[1:] if c.isdigit()])
codes = cnt.index.to_series().apply(ncodes)
flagged = cnt.index[(codes.values != cnt.values)].tolist()
print("이미지당 객체 수:", dict(cnt.value_counts().sort_index()))
print(f"\\n품질플래그 {len(flagged)}장 (파일명 약품 수 ≠ 실제 bbox 수):")
for fn in flagged:
    print(f"  codes={ncodes(fn)}  boxes={cnt[fn]}   {fn}")""")
)

cells.append(
    code("""groups = {fn: g for fn, g in ann.groupby("file_name")}
n = len(flagged); cols = 4; rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 4.4 * rows))
axes = np.array(axes).reshape(-1)
for i, fn in enumerate(flagged):
    ax = axes[i]; img = cv2.imread(str(TRAIN_IMG / fn))
    if img is None:
        ax.set_title("load fail"); ax.axis("off"); continue
    ax.imshow(img[:, :, ::-1])
    for _, r in groups[fn].iterrows():
        x, y, w, h = r.bbox
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="lime", lw=2.2))
    ax.set_title(f"codes {ncodes(fn)} / boxes {cnt[fn]}", fontsize=10); ax.axis("off")
for j in range(n, len(axes)): axes[j].axis("off")
plt.tight_layout(); plt.show()""")
)

cells.append(md("<<CONCLUSION_C>>"))

nb["cells"].extend(cells)
nbp.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("appended", len(cells), "cells | total", len(nb["cells"]))
