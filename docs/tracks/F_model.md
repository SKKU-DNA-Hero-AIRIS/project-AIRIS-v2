# 트랙 F: 추천 모델 (체형 → 자세·계획)

> 2026-09-30 신설. 트랙 C가 맡던 일 중 모델 부분을 떼어 낸다. C는 최적화·데이터셋·세기·시간 확장에 집중한다.

먼저 읽을 것: `docs/tracks/00_common.md`(소유권·수식), `docs/interfaces.md`("회귀 모델"·"계획 모델" 절), `docs/proposals/flow_matching.md`, `docs/plan_extension.md`.

## 역할

게이트 앞에서 체형과 시나리오를 받아 자세(지금)와 계획(확장 후)을 돌려주는 모델을 만든다. 정답은 트랙 C의 최적화가 만든 데이터셋이고, 최종 선택은 항상 시뮬레이터 재채점으로 한다.

## 소유 파일

`airis/model/*`, `scripts/train_*.py`, `scripts/build_pose_knn.py`, `scripts/run_e5_*.py`, `tests/test_pose_flow.py`, `tests/test_model_*.py`, `docs/experiments_model.md`, `docs/proposals/*`

건드리지 않는 것: `airis/optimize/*`(C), `airis/realtime/*`(E), `airis/sim/*`, `configs/`, 공용 파일(`types.py`, `interface.py`, `docs/interfaces.md`). 필요한 변경은 PR 본문에 제안으로 적는다.

## 인수 시점

트랙 C가 혼합 추천 PR(`predict`의 `backend="hybrid"`, kNN 표, 폴백, `run_e5_flow.py --folds`)을 병합한 뒤부터 F가 소유한다. 그 PR이 열려 있는 동안 F는 코드를 고치지 않고 읽기만 한다.

## 현재 기준 (2026-09-30)

- 단일 자세 추천의 기본은 혼합 방식이다: flow matching 샘플 8 + 가까운 체형(kNN)의 최적 자세 8 + 고정 후보 2 → 재채점 → 최고.
- 체형 5-fold 실측(메시판 300행, 1,500/m²), 하위 5% 점수 비율 · 0.95 미만 비율:

| 방법 | 재채점 | 하위 5% | 0.95 미만 |
|---|---|---|---|
| kNN + 고정 후보 | 18 | 0.996 | 1.0% |
| flow + 고정 후보 | 16 | 0.963 | 2.7% |
| flow (후보 16개 데이터) | 16 | 0.950 | 5.0% |
| 봉우리별 회귀 | 2 | 0.948 | 5.7% |
| 고정 후보표만 | 2 | 0.908 | 10.3% |
| 평균 회귀 | 0 | 0.82 | 41~43% |

- 데이터셋: `data/datasets/pose_dataset_mesh_cand.parquet`(본 폴더, git 밖, 300행, 후보 16개). 결과: `outputs/e5cv_20260929/`.
- 산출물 위치: `data/models/` 또는 환경변수 `AIRIS_MODEL_DIR`.

## 단계

### 1. 혼합 추천 인수 확인 (0.5일)
- 병합된 혼합 추천을 본 폴더 산출물로 5-fold 재현한다. 합격 기준: 하위 5% ≥ 0.99, 0.95 미만 ≤ 2%, 불가 0, 체형 하나 응답 1.5 s 이하.
- 결과를 `docs/experiments_model.md`에 옮긴다(위 표 포함). 트랙 C의 `docs/experiments.md`에는 요약 한 줄과 링크만 요청한다.

### 2. 물리 기준이 바뀔 때의 갱신 절차 (0.5일)
- 계수 k 등 상수가 바뀌면 산출물은 무효다(`physics_hash`). 데이터셋 재생성(C) → 재학습 → kNN 표 재생성 → 5-fold 재확인을 한 명령으로 묶는다.

### 3. 계획 모델 준비 (C의 `PlanEncoder`·D의 `evaluate_plan` 병합 전, 1일)
- `PlanSpace` 키 순서·단계 시간 식·대칭 정규화(`00_common.md` 4.7: 계획 전체에 함께 적용, 1단계 yaw만 0~90°, 각도 감기, 동률 규칙)가 C의 `PlanEncoder`와 같은지 대조 테스트를 만든다. 어긋나면 C와 맞춘다.
- `predict_plan`이 `PlanEncoder.clip_plan`(풍량 한도 보수, 쾌적 상한)을 `to_plan` 뒤·재채점 전에 거치게 한다.

### 4. 계획 모델 비교 (계획 데이터셋이 나온 뒤)
- 비교군: 혼합(flow + kNN + 고정 계획), flow 단독, kNN 단독, 고정 계획표, 짧은 국소 탐색. 재채점 횟수를 같게 맞춘다.
- 채택 기준은 자세와 같다(하위 5%, 0.95 미만 비율). 계획에서 flow 단독으로 충분하면 나머지를 뺄지 총괄에 보고한다.

### 5. 체형 오차 견고성
- 체형 5개 값에 ±5% 오차를 넣었을 때 점수 하락폭(카메라 추정 오차 대응).

## 완료 기준

- [ ] 혼합 추천 5-fold 합격 기준 충족, `docs/experiments_model.md` 기록
- [ ] 산출물 갱신 절차 문서화
- [ ] 계획 모델이 `PlanEncoder`와 같은 규격, `clip_plan` 연결
- [ ] 계획 비교 결과와 채택 제안
- [ ] `pytest -q` 전부 통과, skip 증가 없음

## 병합 후 알림

E(추천 화면 연결), C(데이터셋 열 변경이 있으면)에 통합 관리 세션을 거쳐 알린다.
