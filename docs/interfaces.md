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
| `NozzleConfig` | B (`configs/nozzles.yaml`) | A, D | 위치 (M,3), 방향 (M,3), 세기 (M,). 슬롯 제트면 `slot_axis` (M,3), `slot_length` (M,) 추가 (`None`이면 전부 원형; 배열이면 행 단위로 `slot_length[m] > 0`이고 `slot_axis[m] ≠ 0`인 노즐만 슬롯). 고정 입력 |
| `Scenario` | B (`configs/scenarios.yaml`) | C, A, D | 상속 지원 |
| `BodyState` | B (`build_body`) | A, D, E | 패치 + 캡슐 + `capsule_part`(K,) + `patch_capsule`(N,) |
| `EvalResult` | A, D | C, E | `score`는 최적화용, 나머지는 분석용 |

## 함수 계약

### `build_body(body, pose, scenario, patches_per_m2) -> BodyState` (B)

- 시나리오의 `fixed_pose`가 있으면 해당 자세 변수를 덮어쓴다.
- `seat_height_m`이 있으면 골반 높이를 그 값으로 둔다.
- 패치 법선은 바깥 방향 단위 벡터.
- `capsule_part`는 캡슐 부위(-1 = 가림 전용), `patch_capsule`은 패치의 소속 캡슐 인덱스. 둘 다 채운다.

### `velocity_field_per_nozzle(points, nozzle, t, cfg, surface_normals=None) -> (M, P, 3)` (B)

- 노즐별 자유 제트 기여. `velocity_field`는 이것의 합. 시각 `t`는 펄스용. 수식은 `docs/tracks/00_common.md` 4.1 (원형) / 4.1b (슬롯, `nozzle.slot_axis`가 있는 노즐).
- `surface_normals`가 주어지고 충돌 보정이 켜져 있으면 정체점 보정을 적용한다 (2주차 옵션).
- 몸에 의한 가림은 여기서 처리하지 않는다 (평가기 책임).

### `Evaluator.evaluate(pose, nozzle, body, scenario) -> EvalResult` (A, D)

- 결정론: 같은 입력이면 같은 `score`. 난수는 `cfg.simulation.seed`로 고정.
- 부스 밖 자세(패치가 `|y| > width/2` 또는 `z > height`)는 불가: `score = −1.0`, 제거율 0, `extra["infeasible"] = True` (`00_common.md` 5절).
- `score = Σ part_weights · removal_by_part − discomfort_weight · discomfort`
- `discomfort = Σ discomfort_weights[k] · |pose[k] − pose_default[k]| / range[k]`

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

## 시각화용 상태 덤프 (A → E)

- `outputs/<exp_id>/frames/<step>.npz`: `pos (N,3)`, `attached (N,) bool`, `part (N,) int`, `candidate (N,) int`
- 프레임 간격은 `cfg.simulation.dump_every` (기본 10 스텝). 최적화 중에는 끈다.
