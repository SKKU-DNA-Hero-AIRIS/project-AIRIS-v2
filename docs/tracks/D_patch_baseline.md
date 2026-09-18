# 트랙 D: 패치 기준선·점수·정성 테스트

**목표**: 입자 없이 패치별 벽면 전단만으로 제거율을 계산하는 빠른 평가기를 만들고, 점수 계산을 공용 모듈로 제공하며, 시뮬레이터의 방향이 맞는지 확인하는 정성 테스트를 활성화한다. 입자판이 완성되기 전까지 최적화 파이프라인을 끝까지 돌리는 대체재다.

**파일**: `airis/sim/patch_baseline.py`, `airis/sim/scoring.py`, `tests/fakes.py`, `tests/test_sanity_physics.py`, `scripts/compare_evaluators.py`

**의존성**: B의 `build_body`, `velocity_field_per_nozzle`. 병합 전까지 `tests/fakes.py`로 대체.

**공유 수식**: `00_common.md` 4.2, 4.3, 4.4.

---

## 단계 1. 가짜 입력 (`tests/fakes.py`, 2시간)

다른 트랙도 import하므로 가장 먼저 만든다.

```python
def fake_body(arms_up: bool = False, yaw_deg: float = 0.0) -> BodyState
def fake_nozzles() -> NozzleConfig
def fake_velocity_field_per_nozzle(points, nozzle, t, cfg) -> (M, P, 3)
```

- `fake_body`: 몸통 캡슐 (0.6 m, r 0.15) + 머리 구 (r 0.1) + 팔 캡슐 2개 (0.6 m, r 0.04). `arms_up=True`면 팔이 수평, 아니면 몸통 옆에 붙음. `yaw_deg`로 z 축 회전. 패치는 캡슐마다 원통부만 격자 샘플링해 총 200~300개. `patch_capsule`, `capsule_part` 채운다. 부스 중앙에 둔다.
- `fake_nozzles`: 위치 `(1.0, ±0.6, 1.0)`, `(1.0, ±0.6, 1.4)` 4개, 방향 안쪽, 세기 1.
- `fake_velocity_field_per_nozzle`: `00_common.md` 4.1 수식의 최소 구현. B 병합 후 `jet.velocity_field_per_nozzle`로 교체하고 이 함수는 삭제.

## 단계 2. 점수 모듈 (`airis/sim/scoring.py`, 2시간)

A와 C도 import하는 공용 모듈이다. 시그니처를 먼저 고정한다.

```python
def discomfort(pose: PoseParams, scenario: Scenario) -> float
def score(removal_by_part: np.ndarray, pose: PoseParams, scenario: Scenario, physics_cfg: dict) -> tuple[float, float]
    # returns (score, discomfort)
def removal_fraction(tau: np.ndarray, physics_cfg: dict) -> np.ndarray
    # 로그 정규 CDF. tau=0 → 0
def wall_shear(u: np.ndarray, normals: np.ndarray, physics_cfg: dict) -> np.ndarray
    # (P,3), (P,3) → (P,)  4.2 수식
```

`removal_fraction`은 `scipy.stats.norm.cdf` 대신 `scipy.special.erf`로 쓴다 (Taichi 이식 시 A가 참고).

## 단계 3. 가림 판정 (반나절)

패치 `i`와 노즐 `j` 사이 선분이 다른 캡슐과 교차하면 그 노즐의 기여를 0으로 한다.

- 선분 시작점: `patch_pos + δ·normal` (`δ = air.wall_offset_m`). 끝점: 노즐 위치.
- 캡슐 `k`와의 교차: 두 선분(광선, 캡슐 축) 사이 최단 거리 `< r_k`. 선분-선분 최단 거리는 표준 공식 (Ericson, Real-Time Collision Detection 5.1.9)을 numpy로 벡터화한다.
- 자기 캡슐 `patch_capsule[i]`는 제외한다.
- 결과 `visible (M, N) bool`. 형상: `N × M × K` 거리 계산이므로 N=3600, M=16, K=15 → 86만 원소. numpy로 수 ms.

패치 법선과 노즐 방향이 같은 쪽을 향하면(`normal · (nozzle_pos − patch_pos) < 0`) 노즐이 패치 뒷면에 있으므로 가림 판정 전에 바로 0.

## 단계 4. 평가기 (`PatchEvaluator.evaluate`, 반나절)

```
state   = build_body(body, pose, scenario)                       # B 또는 fake
u_mn    = velocity_field_per_nozzle(state.patch_pos + δ·n, nozzle, t=0, cfg,
                                    surface_normals=state.patch_normal)         # (M,N,3), 4.2b 보정 포함
visible = occlusion(state, nozzle)                               # (M,N)
u       = Σ_m u_mn · visible[m]                                  # (N,3)
τ       = wall_shear(u, state.patch_normal, cfg)                 # (N,)
R       = removal_fraction(τ, cfg)                               # (N,)
R_part  = 면적 가중 평균 per part                                 # (5,)
total   = 면적 가중 평균 전체
score, disc = scoring.score(R_part, pose, scenario, cfg)
return EvalResult(score, R_part, total, disc, extra={"tau": τ, "removal": R, "visible_frac": visible.mean()})
```

`extra`에 패치별 값을 넣어 두면 3D 뷰에서 색으로 볼 수 있다.

`batch_evaluate`는 기본 구현(순차)을 그대로 쓴다. 1회 5 ms면 100개가 0.5초라 충분하다.

## 단계 5. 정성 테스트 활성화 (반나절)

`tests/test_sanity_physics.py`의 `skip`을 제거하고 구현한다. fixture는 B 병합 전후를 자동 전환한다.

```python
@pytest.fixture
def make_body():
    try:
        build_body(BodyParams(), PoseParams(), scenarios["default"])
        return lambda pose, scen: build_body(BodyParams(), pose, scen)
    except NotImplementedError:
        return lambda pose, scen: fake_body(arms_up=pose.shoulder_abduction > 60, yaw_deg=pose.torso_yaw)
```

| 테스트 | 구현 |
|---|---|
| 팔 들면 겨드랑이 상승 | `shoulder_abduction` 20 vs 90. `arms`와 `torso_front` 합산 제거율이 오르는지. 가짜 몸에서는 `arms` 부위만 비교 |
| 등지면 정면 하락 | `torso_yaw` 0 vs 180. `torso_front` 제거율이 떨어지고 `torso_back`이 오르는지 |
| 세기 올리면 상승 | `strengths`를 0.5, 1.0, 1.5배로 한 `NozzleConfig`에서 총 제거율이 단조 증가. ("가까우면 상승"은 가는 자유 제트 모델에서 성립하지 않아 교체. PR #11 발견 사항 1) |
| 세기 0 | `strengths = 0` → `total_removal == 0` |
| 결정론 | 같은 입력 두 번 → `score` 완전 일치 (`==`) |

## 단계 6. 성능 (2시간)

`pytest tests/test_sanity_physics.py --durations=5`로 측정. 목표: 패치 3,600 × 노즐 16 × 캡슐 15에서 `evaluate` 5 ms 이하. 넘으면 가림 판정에서 `N×M×K` 배열을 청크로 나누거나 뒷면 노즐을 먼저 걸러 K 루프를 줄인다.

## 단계 7. E2 비교 스크립트 (2주차, A 병합 후)

`scripts/compare_evaluators.py --n 500 --scenario default`:
- 시나리오 범위 안에서 자세 500개를 균등 샘플링.
- `PatchEvaluator`와 `ParticleEvaluator`로 각각 평가.
- Spearman 순위 상관, 산점도 PNG, 상관이 가장 낮은 자세 10개의 부위별 차이 표를 `outputs/e2/`에 저장.
- 0.8 미만이면 차이가 큰 자세를 `plot_body`로 두 평가기의 패치별 값을 나란히 그려 원인을 찾는다.

## 완료 기준

- [ ] `tests/fakes.py`가 있고 다른 트랙이 import해서 쓴다.
- [ ] `scoring.py`의 네 함수가 `00_common.md` 수식과 일치하고 단위 테스트가 있다 (τ=0 → R=0, τ=τ_med → R=0.5, 불편도 기본 자세 → 0).
- [ ] 정성 테스트 5개가 skip 없이 통과한다 (가짜 몸으로도, B 병합 후 실제 몸으로도).
- [ ] `evaluate` 1회 `patches_per_m2=400`(약 1,000 패치, 노즐 16)에서 15 ms 이하. `PatchEvaluator(physics_cfg, patches_per_m2=...)` 인자로 밀도를 고른다 (기본 2000은 검증용, 최적화 루프는 400).
- [ ] `plot_body(state, values=result.extra["removal"])`로 제거율 분포가 그려지고, 팔을 든 자세에서 겨드랑이가 밝아지는 것이 보인다.

## 병합 후 알림

병합되면 C 세션에 "`DummyEvaluator`를 `PatchEvaluator(load_physics())`로 교체하라", A 세션에 "`scoring.py`를 import하고 인라인 점수 계산을 제거하라"고 전달한다.

## 모델 한계 (PR #11·#18 발견 사항, 2026-09-16)

1. **노즐에 가까워져도 총 제거율이 오르지 않는다.** 자유 제트 가우시안 단면 `σ ∝ s`가 가까울수록 좁아져 덮는 면적이 줄고, 로그 정규 제거율은 포화한다. 노즐을 제트 축 방향으로 |y| 0.6 → 0.4 옮기면 총 제거율이 40~55% 떨어졌다. 그래서 E1의 해당 항목을 "세기를 올리면 상승(단조)"으로 바꿨다 (README 6절, PR #16). 충돌 제트 보정(B 단계 8)이나 평면 제트(4.1b)에서 다시 확인한다.
2. **초기 상수(노즐 4 mm, τ_med 0.5 Pa)로는 제거율이 모든 자세에서 사실상 0이었다.** 몸 표면 τ가 τ_med의 약 1/170. PR #10에서 노즐 지름 25 mm, `fabric_roughness_factor` 0.1(실효 τ_med 0.05 Pa, 미보정 작업값·E3 스윕 대상)로 바꿨다. E1 테스트의 제트 덮어쓰기는 그 뒤 제거 대상.
3. **부스 폭이 좁으면 기본 자세도 벽 밖이다.** 어깨 반폭 0.21 + 팔 반지름 0.045 + 벌림 20°면 |y|max ≈ 0.46 m. 통과 폭 0.9 m(퓨리움 추정)에서는 벌림 약 10° 이하만 가능하고, 만세(벌림 150~180°)는 손끝 z ≈ 2.0 m로 천장 높이에 걸린다. 규칙(00_common 5절)은 그대로 두고 기본 자세·기준 자세를 총괄이 재정의한다.
4. **패치판 1회 시간**: 5,217 패치(2000/m²)에서 36 ms, 1,062 패치(400/m²)에서 약 15 ms (build_body 5.5 + 가림 판정 7 + 나머지 2). 순수 numpy 가림 판정은 이 규모에서 바닥이다. 최적화 루프는 400/m²를 쓴다.
