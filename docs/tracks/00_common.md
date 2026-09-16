# 공통 규약: 병렬 트랙 개발

네 트랙(B 마네킹·제트, D 패치 기준선, C 최적화, A 입자)을 서로 기다리지 않고 동시에 개발하기 위한 규약이다. 각 트랙의 세션은 이 문서와 자기 트랙 문서만 읽으면 시작할 수 있어야 한다.

## 1. 파일 소유권

| 트랙 | 수정 가능 | 절대 수정 금지 |
|---|---|---|
| B | `airis/sim/body.py`, `jet.py`, `scenario.py`, `airis/viz/debug3d.py`, `configs/*.yaml`, `tests/test_body.py`, `tests/test_jet.py` | |
| D | `airis/sim/patch_baseline.py`, `airis/sim/scoring.py`, `tests/test_sanity_physics.py`, `tests/fakes.py`, `scripts/compare_evaluators.py` | `configs/` |
| C | `airis/optimize/*`, `scripts/run_optimize.py`, `scripts/run_baselines.py`, `scripts/run_e4.py`, `tests/test_encoding.py`, `tests/test_cmaes_dummy.py` | `configs/` |
| A | `airis/sim/particles.py`, `airis/sim/kernels/*`, `tests/test_particles.py` | `configs/` |
| 전원 | | `airis/sim/types.py`, `airis/sim/interface.py`, `docs/interfaces.md` |

공용 파일을 바꿔야 한다면 코드를 고치지 말고 PR 설명이나 이슈에 "types.py 변경 제안"으로 남긴다. 병합 담당이 별도 PR로 처리한다.

## 2. 세션 격리

세션마다 worktree와 브랜치를 하나씩 쓴다.

```bash
git worktree add ../airis-v2-body      -b feat/body
git worktree add ../airis-v2-patch     -b feat/patch-baseline
git worktree add ../airis-v2-optimize  -b feat/cmaes
git worktree add ../airis-v2-particles -b feat/particles
```

병합 순서는 **B → D → C → A**. 앞 트랙이 병합되면 자기 worktree에서 `git merge main`으로 받고, 가짜 입력을 실제 구현으로 교체한다.

## 3. 가짜 입력 정책

B가 끝나기 전에 D, C, A가 움직이려면 가짜 입력이 필요하다. 가짜는 **`tests/fakes.py` 한 곳**에만 두고(D 소유), 다른 트랙은 여기서 import한다. B 병합 후에도 fakes.py는 단위 테스트용으로 남긴다.

| 가짜 | 내용 | 대체 시점 |
|---|---|---|
| `fake_body(arms_up: bool) -> BodyState` | 몸통 캡슐 1개, 머리 구 1개, 팔 캡슐 2개. 패치 200개 수준 | B의 `build_body` |
| `fake_nozzles() -> NozzleConfig` | 좌우 벽 각 2개, 높이 1.0/1.4 m, 안쪽 향함 | B의 `load_nozzles` |
| `DummyEvaluator` | 자세 벡터와 목표 벡터의 거리에 음수를 취한 점수 | D의 `PatchEvaluator` |

A는 마네킹 대신 원통 캡슐 1개로 시작하고 `fake_body`도 쓸 수 있다.

## 4. 공유 수식 (모든 트랙이 동일하게 구현)

물리 상수 이름은 `configs/physics.yaml`의 키다. 수식이 다르면 D와 A의 점수가 어긋나 E2가 실패한다. 아래가 단일 기준이다.

### 4.1 자유 제트 속도장 (B가 numpy로, A가 Taichi로 구현)

노즐 위치 `n`, 단위 방향 `d`, 지름 `D = jet.nozzle_diameter_m`, 출구 속도 `U0 = jet.exit_velocity_mps × strength`.
점 `p`에 대해 `r = p − n`, 축 방향 거리 `s = r·d`, 반경 거리 `ρ = |r − s·d|`.

```
s ≤ 0                : u = 0
L_c = K·D            , K = jet.decay_constant
U_c(s) = U0                 (s ≤ L_c)
U_c(s) = U0 · K·D / s       (s > L_c)
σ(s)   = 0.5·D/1.177 + jet.halfwidth_spread_rate · s / 1.177
u(p)   = U_c(s) · exp(−ρ² / (2σ²)) · d
```

노즐 M개의 `u`를 벡터로 합산한다. 펄스가 켜져 있으면 `gate(t) = 1 if ((t/period + phase) mod 1) < duty else 0`을 곱한다. 시각 인자가 없는 정상 상태 평가는 `t = 0`, 펄스 무시.

### 4.2 벽면 전단 (D, A)

패치 위치 `x`, 바깥 법선 `n`. 공기 속도는 `x + δ·n` (`δ = air.wall_offset_m`)에서 조회한다.

```
u_t = u − (u·n)·n                    (접선 성분)
τ   = 0.5 · air.density · air.friction_coeff · |u_t|²
```

정체점 보정(충돌 제트)은 2주차 옵션이며 B가 `jet.py`에 넣는다. 켜지면 `velocity_field`가 이미 보정된 값을 돌려주므로 D와 A는 바뀌지 않는다.

### 4.3 제거율 (D는 닫힌 식, A는 입자 통계)

입자의 임계 전단 `τ_c`는 로그 정규 분포: 중앙값 `adhesion.critical_shear_pa_median × adhesion.fabric_roughness_factor`, 로그 표준편차 `adhesion.critical_shear_sigma_log`.

- **D (패치판)**: 패치의 제거율 = 그 패치에서 `τ_c < τ`인 입자 비율 = 로그 정규 CDF
  `R = Φ((ln τ − ln τ_med) / σ_log)`, `τ = 0`이면 `R = 0`.
- **A (입자판)**: 입자별로 `τ_c`를 샘플링하고 `τ > τ_c`이면 이탈. 최종 제거 비율 = 부스 밖으로 나간 입자 / 초기 입자.

A의 결과는 D에 재부착과 시간 효과가 더해진 것이므로 완전히 같지 않지만, E2의 순위 상관 0.8 이상이 목표다.

### 4.4 불편도와 점수 (D가 `scoring.py`에 구현, A와 C는 import)

```
기본 자세   = PoseParams() 기본값
불편도      = Σ_k  c_k · |θ_k − θ_k,기본| / (hi_k − lo_k)
              c_k = scenario.discomfort_weights.get(k, 0), (lo_k, hi_k) = scenario.pose_bounds[k]
score       = Σ_부위 scoring.part_weights[부위] · R_부위  −  scoring.discomfort_weight · 불편도
```

부위별 제거율 `R_부위`는 면적 가중 평균(D) 또는 입자 수 가중(A).

### 4.5 항력 (A만)

```
Re   = air.density · |v_air − v_p| · d / air.viscosity
f    = 1 + 0.15 · Re^0.687                     (Re < 1000)
τ_p  = particles.density_kg_m3 · d² / (18 · air.viscosity · f)
v_p(t+dt) = v_air + (v_p(t) − v_air) · exp(−dt/τ_p) + g·dt
```

지수 완화 형태라 `dt`가 `τ_p`보다 커도 안정하다.

## 5. 좌표계와 부스

- x: 진행 방향, y: 좌우(중심 0), z: 상하(바닥 0). 단위 m, 각도 degree.
- 부스 크기는 `configs/nozzles.yaml`의 `booth`. 입자가 부스 밖으로 나가면 제거 확정.
- 마네킹은 부스 중앙 `(booth.length_m/2, 0, 0)`에 선다.

## 6. 테스트와 PR

- 각 트랙은 자기 테스트 파일만 추가한다. `pytest -q`가 전부 통과해야 PR을 연다.
- PR 설명에 (1) 완료 기준 체크리스트, (2) 가짜 입력을 쓴 곳, (3) 공용 파일 변경 제안이 있으면 그 내용을 적는다.
- 병합 후 다른 worktree에서 `git merge main`을 하고 깨지는 테스트가 있으면 해당 트랙이 고친다.

## 7. 개발 중 성능 설정

- A의 Taichi 테스트는 입자 1,000개, 후보 2개로 돌린다. 2만 × 100은 성능 측정 때만.
- D의 패치 수는 500~2,000개. 1회 평가 5 ms 이하가 목표.
- 여러 세션이 GPU를 공유하므로 A 외의 트랙은 `ti.init`을 호출하지 않는다.
