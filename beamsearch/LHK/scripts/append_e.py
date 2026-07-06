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
    md("""## 🎨 (e) 배경 · 조명 메타 (도메인갭 예비)

**왜 보는가.** 대회 앵커(1)의 **촬영조건 프로파일**(배경색·조명색·방향·카메라각)을 수치화해, 앞서 육안으로 본 **AI Hub 단일(2)** 과의 도메인갭을 가늠한다(H4 색 불안정 · H5 도메인갭). 색이 얼마나 불안정한지, (2)를 그대로 섞어도 되는지, 증강·정규화를 어디에 걸지가 여기서 갈린다.

**무엇을 보는가.** ① 배경색·조명색·알약방향 분포, ② 카메라 각도(위도/경도) 분포, ③ (1)↔(2) 대조 → 결론.""")
)

cells.append(md("### ① 촬영조건 분포 (이미지 232장 단위)"))
cells.append(
    code("""# 이미지 단위 촬영 메타 (file_name 기준 dedup → 232장)
meta_rows = {}
for jf in ann_files:
    d = json.loads(jf.read_text(encoding="utf-8")); im = d["images"][0]
    fn = im["file_name"]
    meta_rows.setdefault(fn, dict(
        back_color=im.get("back_color"), light_color=im.get("light_color"),
        drug_dir=im.get("drug_dir"), camera_la=im.get("camera_la"),
        camera_lo=im.get("camera_lo"), drug_S=im.get("drug_S")))
meta = pd.DataFrame(list(meta_rows.values()))
print("이미지 메타 shape:", meta.shape)
for col in ["back_color", "light_color", "drug_dir", "camera_la", "camera_lo", "drug_S"]:
    vc = meta[col].value_counts(dropna=False)
    print(f"\\n[{col}] 고유값 {meta[col].nunique(dropna=True)}개")
    print(vc.head(10).to_string())""")
)

cells.append(
    code("""cat_cols = ["back_color", "light_color", "drug_dir"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, col in zip(ax, cat_cols):
    vc = meta[col].astype(str).value_counts().head(10)
    a.bar(vc.index, vc.values, color="#4C78A8")
    a.set(title=f"{col} 분포 (이미지 수)"); a.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.show()

# 카메라 각도(위도×경도) 조합
print("카메라 (la, lo) 조합 분포:")
print(meta.groupby(["camera_la", "camera_lo"]).size().sort_values(ascending=False).head(10).to_string())""")
)

cells.append(md("<<CONCLUSION_E>>"))

nb["cells"].extend(cells)
nbp.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("appended", len(cells), "cells | total", len(nb["cells"]))
