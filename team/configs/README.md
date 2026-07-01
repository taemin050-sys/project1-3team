# configs/

실험 config (재현성의 핵심). 실험 매트릭스 E1~E8을 config 단위로 관리. (docs/02 §6)

- 네이밍: `e1_baseline.yaml`, `e3_res1024.yaml`, `smoke.yaml` …
- 포함: `seed`, 입력 해상도, score/NMS threshold, `max_det`, `device: auto`(CUDA/MPS/CPU 자동) 등
- 한 실험 = 한 config. ablation은 "한 번에 한 변수"(docs/03 §6).
