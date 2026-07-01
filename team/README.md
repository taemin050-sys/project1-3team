# team/ — 승격물 + 공통 데이터

마일스톤 심사를 통과한 결과만 올라오는 **캐논 라인**. 공통 데이터(raw/processed)와 승격 파이프라인·서비스·리포트를 담는다.

- `data/` 공통 데이터(raw 미커밋 / processed SSOT)  · `src/` 승격 파이프라인(data·models·inference)
- `service/` 웹 서비스  · `configs/` 승격 config  · `experiments/` decision-log  · `outputs/` · `report/`
- `notebooks/` 승격된 서사 노트북(제출물)

> 초기엔 파이프라인 레퍼런스(E1 베이스라인·증강·서비스)가 시드로 들어 있고, 마일스톤마다 우수안으로 갱신된다.
