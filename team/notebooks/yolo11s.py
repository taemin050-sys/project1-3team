# -*- coding: utf-8 -*-
import os
import glob  # 폴더 자동 탐색을 위한 모듈
from ultralytics import YOLO
from pathlib import Path

# 윈도우 환경 병렬 처리 에러 방어
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================== PATH CONFIG ==============================
# 현재 스크립트(src/yolo11s.py)의 폴더 위치 (beamsearch/JHB/src/)
current_script_dir = Path(__file__).resolve().parent

project_root = current_script_dir.parent.parent.parent

YAML_PATH = str(project_root / "data.yaml")
# =========================================================================

def train_phase1_freeze():
    """
    [Phase 1] Backbone Freeze & Head Training
    목적: Pretrained 지식을 보존한 채, 고해상도(1280px) 입력에 맞춰
          새로운 데이터셋의 Head Layer 안정화한다
    """
    print("\n 🚀 [Phase 1] 시작: Backbone Freeze 초기 학습")

    # 디버그: 실행 시점에 실제로 읽히는 yaml 내용을 그대로 출력
    print("===== YAML_PATH 실제 내용 =====")
    with open(YAML_PATH, encoding="utf-8") as f:
        print(f.read())
    print("================================")

    # Backbone: Pretrained 모델 사용
    model = YOLO('yolo11s.pt')

    # Phase 1의 학습 결과를 변수(results)에 담아 반환합니다.
    results = model.train(
        data=YAML_PATH,
        epochs=30,  # 초기 안정화는 30에폭으로 가볍게 진행
        imgsz=1280,
        freeze=10,  # 초기 Backbone 10개 레이어 동결
        optimizer='AdamW',
        cos_lr=True,
        lr0=1e-3,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        box=12.5,  # Box Regression(CIoU Loss) 가중치 강화
        batch=8,
        patience=10,  # Early Stopping
        project="pill_project",
        name="stage1_freeze",
        workers=4,
        device=0
    )
    print("✅ [Phase 1] 완료!")
    return results


def train_phase2_finetune(phase1_results=None):
    """
    [Phase 2] Full Unfreeze Fine-Tuning
    목적: 전체 레이어의 잠금을 풀고 아주 미세한 학습률로
          정교하게 학습합니다.
    """
    print("\n 🔥 [Phase 2] 시작: Full Unfreeze 미세 조정 (Fine-Tuning)")

    # 가장 최신 best.pt를 찾아옵니다.
    best_model_path = None

    # 1. Phase 1의 결과 객체가 정상적으로 전달되었다면 그 안에서 경로 추출
    if phase1_results is not None and hasattr(phase1_results, 'save_dir'):
        best_model_path = os.path.join(phase1_results.save_dir, "weights", "best.pt")
    else:
        # 2. 만약 직접 추출에 실패하더라도 폴더 생성 시간을 비교해 가장 최근 폴더를 탐색
        search_path = os.path.join("pill_project", "stage1_freeze*")
        folders = glob.glob(search_path)
        if not folders:
            print("❌ [ERROR] Phase 1 학습 폴더를 찾을 수 없습니다. Phase 1이 정상 완료되었는지 확인하세요.")
            return

        # 가장 최근에 수정된 폴더 찾기
        latest_folder = max(folders, key=os.path.getmtime)
        best_model_path = os.path.join(latest_folder, "weights", "best.pt")

    print(f"📦 로딩할 Phase 1 최고 가중치 경로: {best_model_path}")

    if not os.path.exists(best_model_path):
        print(f"❌ [ERROR] 해당 경로에 best.pt 파일이 존재하지 않습니다: {best_model_path}")
        return

    # 찾아낸 가중치로 모델 로드
    model = YOLO(best_model_path)

    model.train(
        data=YAML_PATH,
        epochs=70,
        imgsz=1280,
        freeze=0,  # Full Fine-Tuning
        optimizer='AdamW',
        cos_lr=True,
        lr0=1e-4,
        warmup_epochs=2.0,
        weight_decay=0.0005,
        batch=8,
        patience=20,
        project="pill_project",
        name="stage2_finetune",
        workers=4,
        device=0
    )
    print("🎉 [Phase 2] 완료!")


if __name__ == '__main__':

    phase1_output = train_phase1_freeze()

    train_phase2_finetune(phase1_output)