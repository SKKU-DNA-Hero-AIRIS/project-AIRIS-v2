# 인터페이스 계약

병렬 개발의 전제. 여기 적힌 시그니처와 배열 형태는 1주차 이틀 안에 고정하고, 이후 변경은 PR + 전원 리뷰.
실제 정의는 `airis/sim/types.py`, `airis/sim/interface.py`가 단일 소스이며 이 문서는 설명이다.

## 좌표계와 배열 규약

- x: 게이트 진행 방향, y: 좌우, z: 상하 (바닥 = 0). 단위 m.
- 위치·방향 배열은 `float32 (N, 3)`. 부위 ID는 `int32 (N,)`, `PART_NAMES` 인덱스.
- `PART_NAMES = ["head", "torso_front", "torso_back", "arms", "legs"]`

## 타입 소유자

| 타입 | 생산자 | 소비자 | 비고 |
|---|---|---|---|
| `BodyParams` | E(포즈 추정), C(샘플링) | B | 체형 5개 값. 메시 몸 기준 정의(관절 중심): 키, 좌우 upperarm01 머리 간격, 유두선 높이 몸통 앞뒤 폭, 상완+전완 구간 합(자세 불변), 고관절 높이. 기본값 1.70/0.342/0.194/0.463/0.883 = `human_mesh.MESH_DEFAULT_BODY` (#62, 2026-09-18) (`docs/mesh_transition.md` 8) |
| `PoseParams` | C(최적화 변수) | B | degree. `to_vector / from_vector` 순서 고정 |
| `NozzleConfig` | B (`configs/nozzles.yaml`) | A, D | 위치 (M,3), 방향 (M,3), 세기 (M,). 슬롯 제트면 `slot_axis` (M,3), `slot_length` (M,) 추가 (`None`이면 전부 원형; 배열이면 행 단위로 `slot_length[m] > 0`이고 `slot_axis[m] ≠ 0`인 노즐만 슬롯). 고정 입력 |
| `Scenario` | B (`configs/scenarios.yaml`) | C, A, D | 상속 지원 |
| `BodyState` | B (`build_body`) | A, D, E | 패치 + 캡슐 + `capsule_part`(K,) + `patch_capsule`(N,). 메시 모델이면 `mesh_vertices`(V,3)·`mesh_faces`(F,3)·`mesh_face_part`(F,)·`patch_face`(N,) 추가, `capsules`는 뼈 근사 캡슐 |
| `EvalResult` | A, D | C, E | `score`는 최적화용, 나머지는 분석용 |
| `Plan`, `Phase` | C(최적화 변수), 모델 | A, D, E | 자세 순서(`phases`: 자세 + 시간) + 구역 세기 `zone_strengths` (`ZONE_NAMES` 순서, 5개). `docs/plan_extension.md` |

## 함수 계약

### `build_body(body, pose, scenario, patches_per_m2) -> BodyState` (B)

- 시나리오의 `fixed_pose`가 있으면 해당 자세 변수를 덮어쓴다.
- `seat_height_m`이 있으면 골반 높이를 그 값으로 둔다.
- 패치 법선은 바깥 방향 단위 벡터.
- `capsule_part`는 캡슐 부위(-1 = 가림 전용), `patch_capsule`은 패치의 소속 캡슐 인덱스. 둘 다 채운다.
- `configs/physics.yaml` `body.model`이 `mesh`면 MakeHuman sim 메시(약 5~6k 삼각형)에 자세·체형을 적용해 `mesh_vertices`(부스 좌표, float32)·`mesh_faces`(바깥 방향 반시계)·`mesh_face_part`·`patch_face`를 채우고, `capsules`/`capsule_part`에는 뼈에 맞춘 근사 캡슐(휠체어 프레임 포함)을 담는다. 패치 샘플링 규약은 `docs/mesh_transition.md`.
- 메시 모델의 패치 격자도 y → −y 대칭 쌍이어야 한다 (#42 규칙).

### `airis/sim/human_mesh.py` (B, 메시 몸)

- `load_makehuman(path=None) -> MeshAsset`: 기본 자세 정점 (V,3), 면 (F,3), 뼈(이름·부모·기본 변환), 정점별 뼈 가중치(희소). 자산은 `data/meshes/makehuman/`.
- `pose_mesh(asset, body: BodyParams | None, pose: PoseParams, scenario) -> (V,3)`: 체형(뼈 축척·타깃) + 자세(선형 블렌드 스키닝) 적용, 부스 좌표(발바닥 z=0, 부스 중앙, `seat_height_m` 반영). `body=None`이면 `MESH_DEFAULT_BODY`. 1회 수 ms.
- 패치 면적 = 면 면적 / 기대 패치 수(≈ 1/밀도). sim 메시는 `scripts/build_sim_mesh.py`(B)가 만들고 `data/meshes/makehuman/sim_mesh.npz`에 커밋한다.
- 좌우 대칭 면 인덱스 맵과 면 → 부위 맵은 자산과 함께 저장한다.

### `velocity_field_per_nozzle(points, nozzle, t, cfg, surface_normals=None) -> (M, P, 3)` (B)

- 노즐별 자유 제트 기여. `velocity_field`는 이것의 합. 시각 `t`는 펄스용. 수식은 `docs/tracks/00_common.md` 4.1 (원형) / 4.1b (슬롯, `nozzle.slot_axis`가 있는 노즐).
- `surface_normals`가 주어지고 `jet.impingement.enabled`면 충돌 제트 → 벽면 제트 보정(`00_common.md` 4.2b)을 적용한 값을 돌려준다. D는 항상 `state.patch_normal`을 넘긴다.
- 몸에 의한 가림은 여기서 처리하지 않는다 (평가기 책임).

### `occlusion(state, nozzle, physics_cfg) -> (M, N)` (D)

- 노즐 m의 가림점(슬롯은 축 위 K점)에서 패치 n이 보이는 비율. 캡슐 모델은 선분-캡슐 교차, 메시 모델은 `mesh_vertices`/`mesh_faces` 광선-삼각형 교차(Open3D `RaycastingScene`)로 판정하고 `patch_face`의 자기 면은 제외한다. A는 이 함수를 그대로 재사용한다.

### `Evaluator.evaluate(pose, nozzle, body, scenario) -> EvalResult` (A, D)

- 결정론: 같은 입력이면 같은 `score`. 난수는 `cfg.simulation.seed`로 고정.
- 부스 밖 자세(패치가 `|y| > width/2` 또는 `z > height`)는 불가: `score = −1 − 10·d_out`(벽 초과 거리 m), 제거율 0, `extra["infeasible"] = True` (`00_common.md` 5절). 불가 판정은 `score ≤ −1.0`.
- `score = Σ part_weights · removal_by_part − discomfort_weight · discomfort`
- `discomfort = Σ discomfort_weights[k] · |pose[k] − pose_default[k]| / range[k]`

### `Evaluator.evaluate_plan(plan, nozzle, body, scenario) -> EvalResult` (D, A)

- 계획 평가. 수식은 `00_common.md` 4.4(계획 점수)·4.6(시간)·4.7(세기·에너지). `nozzle`은 기준 배치이고 평가기가 단계마다 `apply_zone_strengths`(`airis.sim.scenario`)를 부른다.
- `extra`: `energy`(무차원 e), `duration_s`, `removal_by_part_per_phase` (K, 5), 불가면 `infeasible`. 어느 단계든 부스 밖이면 계획 전체가 불가(5절 벌점은 단계 중 최대 `d_out`).
- 단계 1개, 구역 세기 전부 1, `adhesion.kinetics.enabled` 거짓이면 `evaluate(pose, …)`와 제거율이 같아야 한다 (회귀 테스트).
- 구현하지 않은 평가기는 `NotImplementedError`.

### `apply_zone_strengths(nozzle, zone_strengths, torso_yaw, zones=None) -> NozzleConfig` (B)

- 위치: `airis.sim.scenario` (`from airis.sim.scenario import apply_zone_strengths`, #88). `airis/sim/__init__.py` 재수출은 없다.
- 구역 세기를 노즐별 `strengths`에 곱한 새 `NozzleConfig` (`s_m = zone_strengths[구역(m)] × nozzle.strengths[m]`). 구역 → 노즐 매핑은 `configs/nozzles.yaml`의 `zones`와 `torso_yaw`(degree, 가슴 쪽 벽 판정, `00_common.md` 4.7)로 정한다.
- `zones`는 구역 정의 `{side_low_z, side_high_z, z_tolerance_m}`. `None`이면 `nozzles.yaml`의 `active` 배치 구역 정의(`zone_config()`)를 쓴다.
- 입력 `nozzle`은 바꾸지 않는다. 한도(`fan.s_max`, 풍량 한도, 쾌적 상한) 보수는 하지 않는다 (C의 인코더 몫). `zone_strengths`가 `ZONE_NAMES` 순서 5개가 아니거나 음수·비유한값이면 `ValueError`. 어느 구역에도 들지 않는 노즐이 있어도 `ValueError`.
- 보조 함수 (같은 모듈, B):
  - `nozzle_zone_index(nozzle, torso_yaw, zones=None) -> (M,)`: 노즐별 구역 번호(`ZONE_NAMES` 인덱스).
  - `zone_nozzle_counts(nozzle=None, zones=None) -> (5,)`: 구역별 노즐 수. 기준 배치 `slot_bars`는 `[2, 2, 2, 2, 4]`. `nozzle`이 `None`이면 `load_nozzles()`. C의 풍량 한도 보수(`Σ_m s_m ≤ cap_ratio · M`)에 쓴다.
  - `chest_wall_sign(torso_yaw) -> int`: 가슴 쪽 벽의 y 부호(+1 = +y 벽, −1 = −y 벽). `sin(yaw) ≥ 0`이면 +1, 정면·후면(`|sin| < 1e−6`, ±180 포함)은 +1로 고정.
  - `zone_strength_caps(scenario) -> (5,)`: 구역별 쾌적 상한(`scenario.nozzle_strength_cap`의 구역 이름 키). 상한이 없는 구역은 `inf`. 키는 구역 이름(`ZONE_NAMES`)이어야 하고, 다른 키(옛 부위 키 `torso_front` 등)는 `ValueError` (`load_scenarios`도 같은 검사를 한다).

### `Evaluator.batch_evaluate(candidates, body, scenario) -> (B,)` (A 오버라이드)

- 입력 순서와 출력 순서가 같아야 한다.
- 입자판은 후보 B개를 한 커널에서 처리. 후보 간 상태가 섞이지 않는지 테스트 필수.

## 최적화 벡터 인코딩 (C)

- `PoseParams` 중 `scenario.fixed_pose`에 없는 변수만 이어붙인다 (기본 7개, 휠체어는 5개).
- 각 변수는 `pose_bounds`로 [-1, 1] 정규화.
- 노즐은 `configs/nozzles.yaml`에서 로드한 고정 `NozzleConfig`를 그대로 넘긴다 (변수 아님).
- README 12절의 노즐 확장을 채택하면 노즐 M개 × (높이 z, yaw, pitch, 세기)가 벡터 뒤에 붙는다. 그 전까지는 구현하지 않는다.

## 데이터셋 스키마 (C → E)

parquet, 한 행 = 체형 1개 × 시나리오 1개.

| 열 | 타입 | 설명 |
|---|---|---|
| `body_height_m` … | float | `BodyParams` 필드 전부 |
| `scenario` | str | |
| `pose_shoulder_abduction` … | float | 최적 `PoseParams` |
| `nozzle_layout_hash` | str | 사용한 고정 노즐 배치의 해시 |
| `score`, `total_removal`, `discomfort` | float | |
| `exp_id`, `commit`, `seed` | str/int | 재현용 |
| `candidate_k` | int | 체형당 저장한 근사 최적 후보 수 설정. 0 이면 아래 열 없음 |
| `cand_pose_*`, `cand_score`, `n_candidates` | list[float] / int | `candidate_k > 0` 일 때만. best 대비 `candidate_tol` 안의 가능한 후보를 점수순·서로 떨어진 것만 최대 k 개 (yaw 접은 값). flow matching 학습 점 (`airis/optimize/dataset.py`) |

## 회귀 모델 (C → E)

`airis/model/predict.py` (C, 4주차). E의 `airis/realtime/recommend.py`가 호출한다. 그 전까지 E는 `docs/experiments.md`의 시나리오별 최적 자세 표를 돌려주는 스텁을 쓴다.

```python
def predict_pose(body: BodyParams, scenario: Scenario) -> PoseParams
```

- 시나리오는 객체로 받고 안에서는 `scenario.name`으로 구분한다 (데이터셋 열 `scenario`가 str). 학습에 없던 이름은 `KeyError`.
- 출력은 항상 `PoseEncoder(scenario).clip_pose()`로 투영한다 → `pose_bounds` 안, `fixed_pose` 적용(휠체어 hip/knee 90).
- **기본 구현: 혼합 방식 (총괄 결정 2026-09-30).** 후보 = flow matching 샘플 8 (`airis/model/flow.py`) + 가까운 체형(kNN)의 최적 자세 8 + 호출자가 넘기는 고정 후보(`extra_candidates`, E의 후보표 2) → `clip_pose` 후 학습 데이터와 같은 몸 모델·밀도의 패치판으로 재채점 → 최고 (`airis/model/predict.py`, `backend="hybrid"`). `Prediction.sources`에 후보별 출처(`flow`·`knn`·`extra`)를 남긴다. 같은 입력이면 같은 출력이다(난수 고정).
- 근거 (체형 5-fold, 메시판 300행, 1,500/m², `outputs/e5cv_20260929`): 하위 5% 점수 비율 · 0.95 미만 비율이 kNN + 고정 후보 0.996 · 1.0%, flow + 고정 후보 0.963 · 2.7%, flow 0.950 · 5.0%, 봉우리별 회귀 0.948 · 5.7%, 고정 후보표만 0.908 · 10.3%, 평균 회귀 0.82 · 41~43%. kNN 단독은 키 1.93 m에서 이웃의 만세가 모두 천장에 걸려 불가가 나오므로 고정 후보를 함께 재채점한다. 합격 기준: 5-fold 하위 5% ≥ 0.99, 0.95 미만 ≤ 2%, 불가 0, 응답 1.5 s 이하.
- 산출물: `data/models/pose_flow.pt`(flow, torch 필요)와 `data/models/pose_knn.parquet`(kNN 표: `body_*`, `scenario`, `pose_*`, `score`, `arm_class`, 해시·몸 모델·밀도·커밋). 한쪽만 있으면 있는 쪽 + 고정 후보로 동작하고 경고를 한 번 낸다. 같은 코드가 출력 공간만 바꿔 계획 모델(`PlanSpace`, 21차원)도 학습한다.
- 산출물 `data/models/pose_flow.pt`(`scripts/train_pose_flow.py`)에 `nozzle_layout_hash`, `physics_hash`, 학습 커밋을 함께 저장하고, 로드 시 현재 설정과 다르면 경고한다. 물리 기준이 바뀌면 모델은 무효다 (`docs/experiments.md`와 같은 규칙).
- **다봉 지형 처리**: 부스·노즐이 좌우 대칭이라 `torso_yaw ±θ`가 동등하다 → 데이터셋 생성(C 단계 9)에서 yaw를 `|yaw|`로 접는다(거울 정규화). **앞뒤 등가(2026-09-21 메시판 E4에서 확인, C)**: 슬롯 배치가 진행 방향(x)으로도 대칭이고 `part_weights`의 torso_front/back이 같아 `yaw θ`와 `180° − θ`의 점수가 같다(0.6533 vs 0.6534) → 한 번 더 `90° − |90° − |yaw||`로 접어 0~90°로 정규화한다. 원래 yaw 열은 데이터셋에 유지한다. 슬롯 배치의 x 대칭이나 front/back 가중치가 달라지면 이 두 번째 접기는 제거한다. 팔 벌림은 "팔 내림"과 "만세" 두 봉우리 사이 평균(≈90°)이 부스 밖일 수 있으므로, 모델은 봉우리를 먼저 분류하고 그 안에서 회귀하거나 최소한 출력 후 `outside_booth`로 부스 안인지 검사해 가까운 봉우리로 투영한다.
- 평가(E5): 예측 자세를 시뮬레이터에 넣은 점수 / 직접 최적화 점수. README H3의 중앙값 95%는 고정 후보표 + 재채점(스텁)이 이미 넘으므로, 채택 기준은 **하위 5% 점수 비율과 0.95 미만 비율**이다. `scripts/run_e5_flow.py`가 flow · flow+stub · flow1(샘플 1개) · kNN + 재채점 · 봉우리별 회귀(clsreg) · HGB·MLP 평균 회귀 · 스텁을 holdout 체형에서 비교하고, 재채점 횟수(`n_evals`)와 경계 구간(wheelchair 키 1.45~1.60 m, 선 자세 키 1.83 m 이상)을 함께 보고한다.
- 예외 규약(E의 `recommend`가 스텁으로 폴백할 때 구분한다): 모듈이 없거나 쓸 수 있는 백엔드가 없으면(`torch` 없음 + kNN 표 없음) `ImportError`, 산출물이 둘 다 없으면 `FileNotFoundError`, 학습에 없던 시나리오면 `KeyError`. 그 외 예외는 삼키지 않는다. `torch`가 없어도 kNN 백엔드는 동작해야 한다.

## 계획 모델 (F → E, 확장; 2026-10-06 갱신, PR #151·#152)

```python
def predict_plan(body: BodyParams, scenario: Scenario, *, rotation_pose: PoseParams | None = None, ...) -> Plan
```

- 구현: `airis/model/predict.py`의 `predict_plan` / `predict_plan_candidates`. **기본은 혼합**(`backend="hybrid"`): flow 샘플 8 (`plan_flow.pt`) + 가까운 체형의 kNN 계획 8 (`plan_knn.parquet`, `PlanKNN`) + 고정 회전 계획 2 (`airis/optimize/baselines.rotation_plan`, 10단계 × 2 s: **제안 자세 그대로 회전**(`rotation_pose`에 자세 추천 결과를 넘길 때) · 기본 자세 회전) → 패치판 `evaluate_plan`으로 재채점 → 최고. `n_samples`만 주는 옛 호출은 flow 단독(고정 계획 없음)으로 그대로 동작한다. 부스 안 판정 여유(`feasibility_margin`, `feasibility_margin_mode all|reach`)는 자세 경로와 같은 의미로 재사용한다(키운 체형에서도 부스 안인 첫 후보, 점수는 추정 체형 것).
- **단계 수 N은 코드에 고정하지 않는다.** 산출물(flow의 출력 공간, kNN 표의 열 수)에서 읽고, 둘이 다르면 `ValueError`. 후보 투영 한도도 설정 파일이 아니라 산출물에서 읽는다(`plan_limits_for`: 총 시간 하한은 N × `min_phase_s` 이상으로 올린다).
- 계획 벡터 키 순서(데이터셋 열 `plan_<키>`, 8N + 5): `p1_<자세 7개>` … `pN_<자세 7개>`, `duration_s`, `share_1` … `share_{N−1}`, `zone_chest_low`, `zone_chest_high`, `zone_back_low`, `zone_back_high`, `zone_top` (`airis/model/flow.PlanSpace.plan_keys`). 단계 시간 = `min_phase_s + (T − N·min_phase_s)·몫`, 마지막 단계가 나머지 몫. 대칭 정규화는 `00_common.md` 4.7(계획 전체에 함께, 1단계 yaw만 0~90°).
- **계획 데이터셋 스키마**(C 생성, `scripts/run_dataset.py` 계획 모드; C·F 합의 2026-10-06): 한 행 = 체형 1개 × 시나리오 1개. 체형은 `body_seed 0`으로 자세 데이터셋과 같은 `body_idx`. 열: `body_*` 5 + `scenario` + `plan_<키>` 8N+5 + `score`·`total_removal`·`energy`·`duration_s`·`discomfort` + 후보 **`cand_plan_raw`**(정규화 전 raw 벡터 목록; `cand_plan_<키>`로 펴지 않는다) · `cand_score` · `n_candidates` + 도장 10개(`physics_hash`·`nozzle_layout_hash`·`body_model`·`patches_per_m2`·`commit`·`kinetics_enabled`·`time_constant_s`·`zone_nozzle_counts`(문자열)·`n_phases`·`energy_weight`) + 한도 6개(`duration_lo_s`·`duration_hi_s`·`min_phase_s`·`transition_s`·`s_max`·`cap_ratio`; 실행이 하한을 올리므로 설정 파일 값과 다를 수 있다) + 선택 열(기준선 점수 `score_p1_10`·`score_p1opt_10`, 단일 자세 최적 `pose_*`). 후보 선정 거리는 차원으로 정규화한다(÷√dim, 임계값은 C가 1행 실측 뒤 확정). 설정 도장 열이 섞이면 거부, `commit`은 경고 뒤 `a+b`로 잇는다(자세 경로와 같은 규칙).
- 산출물: `data/models/plan_flow.pt` + `data/models/plan_knn.parquet`(도장·한도 열 포함). 갱신 명령 `scripts/train_plan_models.py`(학습 → kNN 표 → 체형 5-fold → 판정 → 합격 시 설치, `gate.json`에 설치 이력 누적, 시작 시점 커밋 기록). 평가 `scripts/run_e5_plan.py`(체형 K-fold, `run_e5_flow`와 같은 분할 규칙, 방법: hybrid·flow+fixed·knn+fixed·fixed 등).
- **판정 기준**(완성도 지표 F2·F3): 체형 5-fold에서 추천 계획의 `evaluate_plan` 점수 / 데이터셋 최적 점수의 **하위 5% ≥ 0.97, 불가 0, 평균 응답 ≤ 1.5 s**. 단계 수가 달라도 점수 자는 같다(`score_plan`은 N 무관)지만 총 시간이 다르면 불공정하므로 비교 표에 총 시간을 병기한다.
- **미결(응답 시간)**: 계획 1개 채점이 단계 수에 비례하면 N=9 후보 18개 재채점은 1.5 s를 넘는다. 재채점은 현재 순차. C가 데이터셋 1행 실측 때 `evaluate_plan` 1회 시간(N=5·9)을 재고, 그 값으로 후보 수·병렬 채점(D의 스레드 안전 확인 필요)·목표 완화 중 하나를 총괄이 정한다.
- 예외·해시 규약은 `predict_pose`와 같다. 자세 산출물을 계획 모델로(또는 반대로) 읽으면 `ValueError`. 출력은 `PlanEncoder.clip_plan()`으로 투영한다. 장비 제어 출력(E)은 단계별 실제 팬 값(`apply_zone_strengths(…).strengths`)을 단계 안에 둔다.

## 시각화용 상태 덤프 (A → E)

- `outputs/<exp_id>/frames/<step>.npz`: `pos (N,3)`, `attached (N,) bool`, `part (N,) int`, `candidate (N,) int`
- 프레임 간격은 `cfg.simulation.dump_every` (기본 10 스텝). 최적화 중에는 끈다.
