# 공통 규약: 병렬 트랙 개발

네 트랙(B 마네킹·제트, D 패치 기준선, C 최적화, A 입자)을 서로 기다리지 않고 동시에 개발하기 위한 규약이다. 각 트랙의 세션은 이 문서와 자기 트랙 문서만 읽으면 시작할 수 있어야 한다.

## 1. 파일 소유권

| 트랙 | 수정 가능 | 절대 수정 금지 |
|---|---|---|
| B | `airis/sim/body.py`, `human_mesh.py`, `jet.py`, `scenario.py`, `airis/viz/debug3d.py`, `configs/*.yaml`, `data/meshes/*`, `scripts/build_sim_mesh.py`, `tests/test_body.py`, `tests/test_jet.py`, `tests/test_human_mesh.py` | |
| D | `airis/sim/patch_baseline.py`, `airis/sim/scoring.py`, `tests/test_sanity_physics.py`, `tests/fakes.py`, `scripts/compare_evaluators.py`, `scripts/compare_bodies.py` | `configs/` |
| C | `airis/optimize/*`, `scripts/run_optimize.py`, `scripts/run_baselines.py`, `scripts/run_e4.py`, `scripts/run_e3.py`, `scripts/run_dataset.py`, `docs/experiments.md`, `tests/test_encoding.py`, `tests/test_cmaes_dummy.py` | `configs/` |
| A | `airis/sim/particles.py`, `airis/sim/kernels/*`, `tests/test_particles.py` | `configs/` |
| E | `airis/realtime/*`, `airis/viz/*`(단 `debug3d.py`는 B), `scripts/run_dashboard.py`, `scripts/render_frames.py`, `tests/test_realtime.py`, `tests/test_viz.py`, `docs/figures/*` | `configs/`, `airis/sim/*` |
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

#### 4.1b 슬롯(평면) 제트 — 크로스팬 토출구 (B가 numpy로, A가 Taichi로 구현)

기준 장비(퓨리움 PURIUM-10000-P)는 원형 노즐이 아니라 길이 `L`, 높이 `h`의 가로 슬롯에서 나오는 크로스팬 제트다. 노즐별 판정: `NozzleConfig.slot_axis`가 `None`이면 전부 원형(4.1). 배열이면 **행 단위**로, `slot_length[m] > 0`이고 `slot_axis[m]`이 0벡터가 아니면 슬롯(4.1b), `slot_length[m] == 0` 또는 `slot_axis[m] == 0`이면 그 노즐은 원형(4.1)이다. 두 모델은 같은 배치 안에 섞일 수 있다. B의 `load_nozzles`가 이 규약대로 채우고, D·A는 같은 규약으로 읽는다.

슬롯 중심 `n`, 분사 단위 방향 `d`, 슬롯 길이 방향 단위 벡터 `e` (`e ⊥ d`), 길이 `L = NozzleConfig.slot_length[m]`, 높이 `h = jet.slot.height_m`, 출구 속도 `U0 = jet.slot.exit_velocity_mps × strength`.
점 `p`에 대해 `r = p − n`, `s = r·d`, 슬롯 방향 거리 `ρ_e = r·e`, 슬롯 두께 방향 거리 `ρ_n = |r − s·d − ρ_e·e|`.

```
s ≤ 0                : u = 0
L_c = K_p·h          , K_p = jet.slot.decay_constant
U_c(s) = U0                         (s ≤ L_c)
U_c(s) = U0 · sqrt(K_p·h / s)       (s > L_c)          ← 평면 제트는 1/√s 감쇠
σ(s)   = 0.5·h/1.177 + jet.slot.spread_rate · s / 1.177
ρ_e'   = max(|ρ_e| − L/2, 0)                            ← 슬롯 길이 안에서는 균일, 끝에서 가우시안 감쇠
u(p)   = U_c(s) · exp(−ρ_n² / (2σ²)) · exp(−ρ_e'² / (2σ²)) · d
```

`s = L_c`에서 `U_c`가 연속이다 (`sqrt(K_p·h/L_c) = 1`). 펄스 게이트는 4.1과 같다. 상수 `K_p`(≈ 5.8, Rajaratnam 평면 제트 `U_m/U0 = 2.4·sqrt(h/s)`), `spread_rate`(≈ 0.10)는 `configs/physics.yaml`의 `jet.slot`에만 둔다.

### 4.2 벽면 전단 (D, A)

패치 위치 `x`, 바깥 법선 `n`. 공기 속도는 `x + δ·n` (`δ = air.wall_offset_m`)에서 조회한다.

```
u_t = u − (u·n)·n                    (접선 성분)
τ   = 0.5 · air.density · air.friction_coeff · |u_t|²
```

#### 4.2b 충돌 제트 → 벽면 제트 보정 (B가 `jet.py`에, A가 Taichi로 동일 구현. D는 법선만 넘긴다)

4.2만 쓰면 제트를 정면으로 받는 면은 접선 성분이 0이라 τ ≈ 0이 되어, "노즐을 마주 보면 정면 제거율이 오른다"가 뒤집힌다 (PR #11, #24 관찰). 실제로는 정면으로 부딪힌 공기가 정체점에서 방사상으로 퍼지는 벽면 제트가 되어 그 둘레를 문지른다. `jet.impingement.enabled`가 true이고 `surface_normals`가 주어지면 `velocity_field_per_nozzle`이 노즐 m마다 아래 `w·e_r`을 4.1/4.1b의 `u`에 더한 값을 돌려준다. τ는 그 값으로 4.2 그대로 계산한다.

조회점 `x`(= 패치 위치 + δ·n), 바깥 법선 `n`, 노즐 위치 `n_m`, 분사 단위 방향 `d`:

```
cosθ = max(−d·n, 0)                        정면으로 받는 정도. 0이면 보정 없음
H    = ((x − n_m)·n) / (d·n)               노즐에서 접평면까지 제트 축 거리. H ≤ 0이면 보정 없음
c    = n_m + H·d                            충돌점
r    = (x − c) − ((x − c)·n)·n              접평면 위, 충돌점 기준 벡터
       슬롯이면 r ← r − (r·e_t)·e_t          e_t = 슬롯 축 e의 접평면 성분 단위벡터 (선 충돌)
ξ    = |r| / σ(H)                            σ는 4.1/4.1b의 σ(s)를 s = H에서
U_H  = U_c(H)                                4.1/4.1b의 중심 속도를 s = H에서 (strength 포함)
F    = (1 − exp(−ξ²/2)) / ξ                  원형 노즐
F    = (1 − exp(−ξ²/2)) / sqrt(ξ)            슬롯. 슬롯 끝 밖은 exp(−ρ_e'²/(2σ²))를 곱한다
w    = k · cosθ · U_H · F · gate(t)          k = jet.impingement.wall_jet_gain
u_corr = u + w · e_r,   e_r = r / |r|        ξ < 1e-6이면 w = 0
```

- 정체점(ξ = 0)에서 0, 그 둘레 고리에서 최대, 바깥은 원형 1/ξ · 슬롯 1/√ξ로 감쇠한다 (방사상 / 평면 벽면 제트). 봉우리 위치는 Beltaos & Rajaratnam 1974(r/H ≈ 0.14)와 같은 함수족이지만 원문 계수는 미확인이라 `k`를 문헌값으로 고정하지 않는다.
- 설정: `jet.impingement.enabled` (기본 true), `jet.impingement.wall_jet_gain` (기본 1.0). `stagnation_radius_factor`, `wall_jet_start_factor`는 삭제. E3에서 `enabled` 켬/끔과 `k ∈ {0.5, 1, 2}`를 스윕한다.
- 계약: D는 `velocity_field_per_nozzle(probe, nozzle, 0, cfg, surface_normals=state.patch_normal)`로 법선을 넘긴다. A는 부착 입자(패치 법선 있음)에만 적용하고 부유 입자는 자유 제트 그대로. 보정을 켜면 τ가 커지므로 `adhesion.fabric_roughness_factor`를 B가 같은 PR에서 다시 잡는다 (기준 자세 전신 R 5~10% 유지).

**패치판 등가 관계 (C, 2026-09-18, 테스트로 고정)**: 4.2 τ = ½ρ·Cf·|u_t|² 이고 4.3 제거율이 τ/τ_med 에만 의존하며 u 가 출구 속도 U0 에 선형이므로, 패치판에서는 `air.friction_coeff × k ≡ fabric_roughness_factor × 1/k`, `jet.slot.exit_velocity × k ≡ fabric_roughness_factor × 1/k²` 가 수치적으로 같다(차이 1e−8). 따라서 민감도 스윕(E3)은 세 상수 중 거칠기 하나만 돌린다. 입자판은 속도가 입자 수송·항력에도 들어가므로 이 등가가 성립하지 않는다.

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
f    = 1 + 0.15 · Re^0.687                     (Re < 1000, Schiller–Naumann)
f    = 0.0183 · Re                             (Re ≥ 1000, 뉴턴 영역 C_d = 0.44 → f = C_d·Re/24)
τ_p  = particles.density_kg_m3 · d² / (18 · air.viscosity · f)
v_p(t+dt) = v_air + (v_p(t) − v_air) · exp(−dt/τ_p) + g·dt
```

지수 완화 형태라 `dt`가 `τ_p`보다 커도 안정하다. 중앙값 20 µm 입자는 `Re ≈ 30`이라 뉴턴 영역에 들어가지 않으며, 로그 정규 꼬리의 수백 µm 입자만 해당한다. 두 식은 `Re = 1000`에서 연속이다 (`1 + 0.15·1000^0.687 ≈ 18.3 = 0.0183·1000`).

## 5. 좌표계와 부스

- x: 진행 방향, y: 좌우(중심 0), z: 상하(바닥 0). 단위 m, 각도 degree.
- 부스 크기는 `configs/nozzles.yaml`의 `booth`. 입자가 부스 밖으로 나가면 제거 확정.
- 마네킹은 부스 중앙 `(booth.length_m/2, 0, 0)`에 선다.
- **몸 모델**: `configs/physics.yaml` `body.model`이 `mesh`면 MakeHuman 사람 메시(`airis/sim/human_mesh.py`, `docs/mesh_transition.md`), `capsule`이면 캡슐 마네킹. 두 모델 모두 `BodyState`의 패치 필드는 같고, 메시 모델은 `mesh_*`·`patch_face`를 추가로 채우며 `capsules`에는 뼈에 맞춘 근사 캡슐을 담는다. 가림은 메시 모델에서 광선-삼각형(D), 입자 충돌은 근사 캡슐(A).
- **부스 밖 자세는 불가.** `BodyState.patch_pos`가 하나라도 `|y| > booth.width_m/2` 또는 `z > booth.height_m`이면 (x 방향은 열린 문이라 허용) 그 후보는 평가하지 않고 `removal_by_part = 0`, `total_removal = 0`, `extra["infeasible"] = True`, 그리고 **벽을 넘은 거리에 비례한 벌점** `score = −1 − 10·d_out`을 돌려준다. `d_out = max(0, max_i(|y_i| − width/2), max_i(z_i − height))` (m). 조금 닿으면 −1에 가깝고 많이 닿을수록 낮아져 CMA-ES가 부스 안으로 돌아올 기울기를 얻는다. 계수 10 /m은 고정이다. C는 `score ≤ −1.0` 또는 `extra["infeasible"]`로 불가를 판정한다. D와 A가 동일하게 구현한다. 사람마다 체형이 달라 `pose_bounds`로는 막을 수 없고, 벽 밖으로 나간 팔의 먼지가 공짜로 제거되는 것을 막기 위한 규칙이다.

## 6. 테스트와 PR

- 각 트랙은 자기 테스트 파일만 추가한다. `pytest -q`가 전부 통과해야 PR을 연다.
- PR 설명에 (1) 완료 기준 체크리스트, (2) 가짜 입력을 쓴 곳, (3) 공용 파일 변경 제안이 있으면 그 내용을 적는다.
- 병합 후 다른 worktree에서 `git merge main`을 하고 깨지는 테스트가 있으면 해당 트랙이 고친다.

## 7. 개발 중 성능 설정

- A의 Taichi 테스트는 입자 1,000개, 후보 2개로 돌린다. 2만 × 100은 성능 측정 때만.
- D의 패치 밀도는 최적화 400/m²(약 1,000개), 재채점 2,000/m². 1회 평가 중앙값(저부하) 캡슐 모델 30 ms, 메시 모델 100 ms 이하가 기준 (총괄 결정, `docs/mesh_transition.md` 7).
- 여러 세션이 GPU를 공유하므로 A 외의 트랙은 `ti.init`을 호출하지 않는다.
