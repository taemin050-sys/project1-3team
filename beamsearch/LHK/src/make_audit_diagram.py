"""스크럼 공유용: 라벨 자동정리(오류 의심 추출) 파이프라인 다이어그램. 텍스트=PIL 한글."""

import os
import sys
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

OUT = paths.LHK / "label_audit/audit_pipeline_diagram.png"
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"


def f(sz, bold=False):
    return ImageFont.truetype(FONT, sz, index=1 if bold else 0)


W, H = 1720, 1040
BG = (18, 18, 22)
CARD = (34, 36, 44)
BLUE, GREEN, ORANGE, RED = (
    (70, 150, 240),
    (90, 200, 130),
    (255, 190, 80),
    (240, 100, 100),
)
WHITE, GREY = (238, 238, 240), (165, 165, 172)
img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def card(x, y, w, h, fill=CARD, r=14, outline=None, ow=2):
    d.rounded_rectangle(
        [x, y, x + w, y + h], radius=r, fill=fill, outline=outline, width=ow
    )


def ctext(cx, y, s, font, fill=WHITE):
    d.text((cx - d.textlength(s, font=font) / 2, y), s, font=font, fill=fill)


def wrap(x, y, s, font, fill, cx=None, lh=22, maxw=999):
    line = ""
    for ch in s:
        if d.textlength(line + ch, font=font) > maxw:
            (
                ctext(cx, y, line, font, fill)
                if cx
                else d.text((x, y), line, font=font, fill=fill)
            )
            y += lh
            line = ch
        else:
            line += ch
    ctext(cx, y, line, font, fill) if cx else d.text((x, y), line, font=font, fill=fill)
    return y + lh


# 헤더
d.text(
    (40, 26), "라벨 자동정리 — 오류 의심 추출 파이프라인", font=f(34, True), fill=WHITE
)
d.text(
    (42, 72),
    "핵심: '모델 불일치' + '기하 휴리스틱' 2트랙 신호를 가중합해 의심 라벨을 랭킹 → 사람은 상위만 수분 검수",
    font=f(17),
    fill=GREY,
)

# 6단계 플로우
stages = [
    ("① 대상 선정", "real232(교정) + aihub7836\nsynth 제외(=프로그램 라벨)", BLUE),
    ("② 강한 모델 예측", "fold0 최고모델(0.985)로\n각 이미지 검출", BLUE),
    ("③ GT ↔ 예측 매칭", "IoU 최적매칭으로\nGT박스-검출 짝짓기", GREEN),
    ("④ 2트랙 플래그", "모델불일치 + 박스형태\n이상 규칙 부여", ORANGE),
    ("⑤ 가중합 · 랭킹", "플래그 가중치 합산\n= 의심점수 내림차순", ORANGE),
    ("⑥ 출력", "suspects.csv +\n컨택트시트(상위60)", RED),
]
n = len(stages)
bw, bh, gap = 240, 96, 16
x0 = (W - (n * bw + (n - 1) * gap)) // 2
y0 = 118
for i, (t, sub, col) in enumerate(stages):
    x = x0 + i * (bw + gap)
    card(x, y0, bw, bh, outline=col, ow=2)
    ctext(x + bw / 2, y0 + 12, t, f(18, True), col)
    yy = y0 + 40
    for ln in sub.split("\n"):
        ctext(x + bw / 2, yy, ln, f(14), WHITE)
        yy += 20
    if i < n - 1:
        ax = x + bw + 2
        d.line([ax, y0 + bh / 2, ax + gap - 4, y0 + bh / 2], fill=GREY, width=3)
        d.polygon(
            [
                (ax + gap - 4, y0 + bh / 2 - 5),
                (ax + gap - 4, y0 + bh / 2 + 5),
                (ax + gap + 2, y0 + bh / 2),
            ],
            fill=GREY,
        )

# 좌: 2트랙 신호
py = 258
card(40, py, 700, 300, outline=ORANGE, ow=2)
d.text((60, py + 14), "2트랙 신호 (왜 두 개인가)", font=f(20, True), fill=ORANGE)
# 트랙 A
card(60, py + 54, 320, 226, fill=(30, 40, 56))
d.text((78, py + 66), "A. 모델 불일치", font=f(17, True), fill=BLUE)
for i, s in enumerate(
    [
        "강한 모델이 '자신있게' GT와",
        "다르게 보면 라벨을 의심",
        "(Confident Learning 계열)",
        "",
        "· 매칭 검출 없음 → 위치오류",
        "· 클래스 다름 → 라벨오류",
        "· 고신뢰 검출인데 GT無 → 누락",
    ]
):
    d.text(
        (78, py + 92 + i * 26),
        s,
        font=f(14),
        fill=WHITE if s and not s.startswith("(") else GREY,
    )
# 트랙 B
card(400, py + 54, 320, 226, fill=(30, 48, 40))
d.text((418, py + 66), "B. 기하 휴리스틱", font=f(17, True), fill=GREEN)
for i, s in enumerate(
    [
        "모델 없이 박스 '형태'만으로",
        "명백한 오류를 규칙 검출",
        "",
        "· 퇴화(점 박스) / 경계 이탈",
        "· 종횡비 극단(>6, <1/6)",
        "· 클래스 중앙면적 대비 4배↑/¼↓",
        "→ 모델과 상보적 (오탐 보정)",
    ]
):
    d.text(
        (418, py + 92 + i * 26),
        s,
        font=f(14),
        fill=WHITE
        if s and not s.startswith("→")
        else GREEN
        if s.startswith("→")
        else GREY,
    )

# 우: 스코어링 규칙 표
card(760, py, 920, 300, outline=RED, ow=2)
d.text(
    (780, py + 14), "스코어링 규칙 (플래그 · 조건 · 가중치)", font=f(20, True), fill=RED
)
tbl = [
    ("no_pred_match(위치오류)", "매칭 IoU < 0.3", "3.0"),
    ("degenerate(퇴화박스)", "폭/높이 ≤ 2px", "3.0"),
    ("out_of_bounds(경계이탈)", "이미지 밖 좌표", "2.5"),
    ("class_mismatch(클래스오류)", "매칭됐으나 클래스 불일치", "2.5"),
    ("missing_gt(라벨누락)", "고신뢰(>0.6) 검출인데 GT 없음", "2.0×conf"),
    ("size_outlier(크기이상)", "클래스 중앙면적 4배↑ / ¼↓", "1.5"),
    ("extreme_aspect(비율이상)", "종횡비 > 6 또는 < 1/6", "1.5"),
    ("loose_bbox(박스느슨)", "매칭되나 IoU 0.3~0.6", "1.0"),
]
ry = py + 52
d.text((780, ry), "플래그", font=f(14, True), fill=GREY)
d.text((1230, ry), "조건", font=f(14, True), fill=GREY)
d.text((1600, ry), "가중", font=f(14, True), fill=GREY)
ry += 26
for name, cond, wgt in tbl:
    d.text((780, ry), name, font=f(15), fill=WHITE)
    d.text((1230, ry), cond, font=f(14), fill=GREY)
    d.text((1600, ry), wgt, font=f(15, True), fill=ORANGE)
    ry += 28
d.text(
    (780, ry + 2),
    "여러 플래그 동시 해당 → 점수 급상승 → 진짜 나쁜 라벨이 상위로 (사람은 상위만 검수)",
    font=f(14),
    fill=(150, 200, 255),
)

# 하단: 검증 결과 스트립
fy = 588
card(40, fy, 1640, 88, fill=(28, 44, 34), outline=GREEN, ow=2)
d.text((60, fy + 14), "검증", font=f(20, True), fill=GREEN)
d.text(
    (60, fy + 50),
    "246장(3%) 자동 검출 — 팀원 수작업 ~300장과 동급 규모, 사람 개입 0   ·   fold0 제거실험: 0.9833 → 0.9860 (+0.0027)   ·   문제는 aihub 소스 라벨에 집중(240/246)",
    font=f(16),
    fill=WHITE,
)

# 설계 판단 + 한계
gy = 700
card(40, gy, 810, 300, outline=BLUE, ow=2)
d.text((60, gy + 14), "핵심 설계 판단 (왜 이렇게)", font=f(20, True), fill=BLUE)
for i, s in enumerate(
    [
        "· 강한 모델 사용 → 불일치가 '라벨 오류'임을 신뢰(약한 모델이면 모델부족과 혼동)",
        "· 가중합 → 단일 신호 오탐 완충, 다중 신호 겹칠수록 확신↑",
        "· synth 제외 → 프로그램 생성 라벨은 정확, 감사 대상은 외부(aihub)·수기(real)",
        "· val 안전 → 제거는 train 라벨만, holdout 불변 → A/B 공정 보장",
        "· 완전 자동 → 사람은 랭킹 상위만 수분 확인(수일→수분)",
    ]
):
    d.text((60, gy + 52 + i * 30), s, font=f(15), fill=WHITE)

card(870, gy, 810, 300, outline=ORANGE, ow=2)
d.text((890, gy + 14), "한계 · 확장", font=f(20, True), fill=ORANGE)
for i, s in enumerate(
    [
        "· 현재 모델은 대상 이미지를 학습에 포함(in-sample) → 나쁜 라벨 일부 암기 가능",
        "   → 확장: OOF(교차검증 예측)로 자기예측 편향 제거",
        "· 희소·미학습 클래스는 과탐 가능 → 상위 랭킹만 채택으로 완화",
        "   → 확장: Cleanlab(confident learning) 정식 결합으로 확률 보정",
        "· 최종: 팀원 수작업 목록과 교차검증 → 제거 리스트 확정 → 재학습",
    ]
):
    col = (150, 200, 255) if s.strip().startswith("→") else WHITE
    d.text((890, gy + 52 + i * 30), s, font=f(15), fill=col)

img.save(OUT)
print(f">>> 저장: {OUT} ({W}x{H})", flush=True)
