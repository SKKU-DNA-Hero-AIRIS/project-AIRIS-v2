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
| `BodyParams` | E(포즈 추정), C(샘플링) | B | 체형 5개 값 |
| `PoseParams` | C(최적화 변수) | B | degree. `to_vector / from_vector` 순서 고정 |
| `NozzleConfig` | C(최적화 변수) | B, A, D | 위치 (M,3), 방향 (M,3), 세기 (M,) |
| `Scenario` | B (`configs/scenarios.yaml`) | C, A, D | 상속 지원 |
| `BodyState` | B (`build_body`) | A, D, E | 패치 + 캡슐 |
| `EvalResult` | A, D | C, E | `score`는 최적화용, 나머지는 분석용 |

## 함수 계약

### `build_body(body, pose, scenario, patches_per_m2) -> BodyState` (B)

- 시나리오의 `fixed_pose`가 있으면 해당 자세 변수를 덮어쓴다.
- `seat_height_m`이 있으면 골반 높이를 그 값으로 둔다.
- 패치 법선은 바깥 방향 단위 벡터.

### `velocity_field(points, nozzle, t, cfg) -> (P, 3)` (B)

- 자유 제트 합산 + 충돌 보정. 시각 `t`는 펄스용.
- 몸에 의한 가림은 여기서 처리하지 않는다 (평가기 책임).

### `Evaluator.evaluate(pose, nozzle, body, scenario) -> EvalResult` (A, D)

- 결정론: 같은 입력이면 같은 `score`. 난수는 `cfg.simulation.seed`로 고정.
- `score = Σ part_weights · removal_by_part − discomfort_weight · discomfort`
- `discomfort = Σ discomfort_weights[k] · |pose[k] − pose_default[k]| / range[k]`

### `Evaluator.batch_evaluate(candidates, body, scenario) -> (B,)` (A 오버라이드)

- 입력 순서와 출력 순서가 같아야 한다.
- 입자판은 후보 B개를 한 커널에서 처리. 후보 간 상태가 섞이지 않는지 테스트 필수.

## 최적화 벡터 인코딩 (C)

- `PoseParams` 중 `scenario.fixed_pose`에 없는 변수 + 노즐 M개 × (높이 z, 방향 yaw, 방향 pitch, 세기) 를 이어붙임.
- 각 변수는 `pose_bounds` / `nozzle_height_range_m` / [0, 1] 로 [-1, 1] 정규화.
- 노즐의 x, y 위치는 부스 벽면에 고정 (변수 아님). 높이와 방향만 최적화.

## 데이터셋 스키마 (C → E)

parquet, 한 행 = 체형 1개 × 시나리오 1개.

| 열 | 타입 | 설명 |
|---|---|---|
| `body_height_m` … | float | `BodyParams` 필드 전부 |
| `scenario` | str | |
| `pose_shoulder_abduction` … | float | 최적 `PoseParams` |
| `nozzle_0_z`, `nozzle_0_yaw`, … | float | 최적 노즐 벡터 평탄화 |
| `score`, `total_removal`, `discomfort` | float | |
| `exp_id`, `commit`, `seed` | str/int | 재현용 |

## 시각화용 상태 덤프 (A → E)

- `outputs/<exp_id>/frames/<step>.npz`: `pos (N,3)`, `attached (N,) bool`, `part (N,) int`, `candidate (N,) int`
- 프레임 간격은 `cfg.simulation.dump_every` (기본 10 스텝). 최적화 중에는 끈다.
