# 트랙 A: 입자 시뮬레이터 (Taichi GPU)

**목표**: 후보 자세 여러 개를 한 커널에서 동시에 평가하는 입자 기반 평가기. 마네킹 패치 위에 먼지 입자를 뿌리고, 제트가 만드는 전단으로 이탈시키고, 기류를 따라 움직이다 몸에 재부착하거나 부스 밖으로 나가는 과정을 계산한다. 가장 오래 걸리는 트랙이며 가장 마지막에 병합된다.

**파일**: `airis/sim/particles.py`, `airis/sim/kernels/` (필요 시 분리), `tests/test_particles.py`

**의존성**: B의 `build_body`(호스트에서 호출), `00_common.md` 4.1 제트 수식(Taichi로 재구현), D의 `scoring.py`(병합 후 import). 병합 전에는 원통 캡슐 1개와 `tests/fakes.py`로 개발.

**공유 수식**: `00_common.md` 4.1, 4.2, 4.3, 4.5.

---

## 단계 1. 환경 확인 (첫날 오전)

```python
import taichi as ti
for arch in (ti.cuda, ti.vulkan, ti.cpu):
    try:
        ti.init(arch=arch); print("ok", arch); break
    except Exception as e:
        print("fail", arch, e)
```

- 어느 백엔드가 잡히는지 `docs/tracks/A_particles.md` 하단 "환경 기록"에 적는다.
- CUDA가 없으면 Vulkan으로 개발한다. 성능 측정만 클라우드에서.
- 간단한 벤치: `1e7` 원소 SAXPY 커널 실행 시간. 1 ms 이하면 GPU가 잡힌 것.

`ParticleEvaluator.__init__`에서 `ti.init`을 호출하되, 이미 초기화되어 있으면 건너뛴다. 프로세스당 한 번만.

## 단계 2. 데이터 레이아웃 (반나절)

후보 `B`개 × 입자 `N`개를 평탄화해 `B·N` 길이 필드로 둔다. 후보 인덱스는 `i // N`.

| 필드 | 타입 | 형상 | 설명 |
|---|---|---|---|
| `pos` | vec3 f32 | (B·N) | 위치 |
| `vel` | vec3 f32 | (B·N) | 속도 |
| `diam` | f32 | (B·N) | 지름 m |
| `tau_crit` | f32 | (B·N) | 임계 전단 Pa |
| `state` | i32 | (B·N) | 0 부착, 1 부유, 2 제거 |
| `part` | i32 | (B·N) | 부위 인덱스 |
| `normal` | vec3 f32 | (B·N) | 부착 중인 패치의 법선 (전단 계산용) |
| `capsules` | f32 | (B, K_max, 7) | 후보별 캡슐. 후보마다 K가 다르면 패딩 + `n_caps (B)` |
| `cap_part` | i32 | (B, K_max) | 캡슐 부위, −1은 가림 전용 |
| `noz_pos`, `noz_dir` | vec3 f32 | (M) | 노즐, 후보 공통 |
| `noz_strength` | f32 | (M) | |
| `count_init`, `count_removed` | i32 | (B, 5) | 부위별 집계 |

`B`와 `N`은 `__init__(max_candidates, particles_per_candidate)`에서 고정 할당하고 재사용한다. 매 호출 재할당은 느리다.

## 단계 3. 호스트 측 초기화 (반나절)

난수는 **호스트에서 numpy로** 만들어 필드에 복사한다. GPU 난수는 백엔드마다 달라 결정론이 깨진다.

`_init_candidate(b, state: BodyState, rng)`:
1. 패치 면적 누적합으로 입자 `N`개를 패치에 배분 (`rng.random(N)`을 누적합에 `searchsorted`). 면적 비례.
2. `pos = patch_pos[idx] + 1e-4·normal[idx]`, `normal = patch_normal[idx]`, `part = patch_part[idx]`.
3. `diam = lognormal(ln(median_um·1e-6), sigma_log)`, `tau_crit = lognormal(ln(τ_med·roughness), sigma_log)`.
4. `state = 0`, `vel = 0`.
5. 캡슐과 `cap_part` 복사, `count_init[b, part]` 집계.

`rng = np.random.default_rng(seed + b)`. 후보 `b`마다 다른 시드지만 같은 입력이면 같은 결과.

## 단계 4. 제트 속도장 커널 (반나절)

`@ti.func jet_velocity(p: vec3, t: f32) -> vec3`: `00_common.md` 4.1을 그대로. 노즐 M개 루프, 합산.

**검증**: 무작위 점 1,000개에서 Taichi 결과와 `tests/fakes.py`의 `fake_velocity_field_per_nozzle` 합(B 병합 후에는 `jet.velocity_field`)을 비교, 상대 오차 1e-4 이내. 이 테스트가 D와 A의 점수를 맞추는 핵심이다.

## 단계 5. 이탈 판정 (2시간)

부착 입자(`state == 0`)에 대해 매 스텝:

```
u   = jet_velocity(pos + δ·normal, t)
u_t = u − (u·normal)·normal
τ   = 0.5·ρ_air·Cf·|u_t|²
if τ > tau_crit: state = 1, vel = u_t·0.1 + normal·0.05   # 작은 초기 속도
```

`δ = air.wall_offset_m`. 이탈 후에는 `tau_crit`를 재사용하지 않는다 (재부착 시 재샘플링 필요 → 호스트 난수를 미리 `tau_crit_respawn (B·N)`으로 하나 더 준비).

## 단계 6. 부유 입자 적분 (반나절)

`00_common.md` 4.5 지수 완화:

```
u     = jet_velocity(pos, t)
v_rel = |u − vel|
Re    = ρ_air·v_rel·diam/μ
f     = 1 + 0.15·Re^0.687
τ_p   = ρ_p·diam²/(18·μ·f)
vel   = u + (vel − u)·exp(−dt/τ_p) + g·dt
pos  += vel·dt
```

`g = (0, 0, −9.81)`. 20 µm 입자는 `τ_p ≈ 1 ms`라 한 스텝에 거의 기류 속도로 수렴한다. 정상이다.

## 단계 7. 캡슐 충돌과 재부착 (반나절)

부유 입자마다 자기 후보의 캡슐 `K` 루프:

```
c   = 축 선분 위 최근접점(pos)
d   = |pos − c|
if d < r:
    n_hit = (pos − c)/d
    pos   = c + n_hit·(r + 1e-4)                        # 표면 밖으로
    if cap_part ≥ 0 and rand_redep[i] < p_redep:        # 몸에 재부착
        state = 0; part = cap_part; normal = n_hit; tau_crit = tau_crit_respawn[i]; vel = 0
    else:                                               # 반사
        v_n = (vel·n_hit)·n_hit; vel = (vel − v_n)·0.8 − v_n·0.3
```

`rand_redep (B·N)`도 호스트에서 미리 생성. 재부착으로 `part`가 바뀌므로 `count_init`은 초기값 그대로 두고 최종 집계는 `state == 2`인 입자의 **초기** 부위로 한다 → `part_init (B·N)` 필드를 따로 둔다.

## 단계 8. 제거 판정과 집계 (1시간)

- 부스 밖: `pos.x < 0 or pos.x > L or |pos.y| > W/2 or pos.z < 0 or pos.z > H` → `state = 2`. 부스 치수는 `nozzles.yaml booth`에서.
- 시뮬레이션 종료 후 커널로 `count_removed[b, part_init[i]] += 1` (atomic add, 종료 시 1회라 비용 무시).
- `removal_by_part[b] = count_removed / max(count_init, 1)`, `total = Σ removed / N`.
- `score`는 호스트에서 `scoring.score(...)` (D 병합 전 인라인 구현, 병합 후 교체).

## 단계 9. 스텝 루프와 `batch_evaluate` (반나절)

```python
def batch_evaluate(self, candidates, body, scenario):
    B = len(candidates); assert B <= self.max_candidates
    for b, (pose, nozzle) in enumerate(candidates):
        state = build_body(body, pose, scenario)          # 호스트, ~10 ms × B
        self._init_candidate(b, state, rng)
    self._upload_nozzles(candidates[0][1])                # 노즐은 후보 공통
    n_steps = int(duration_s / dt_s)
    for step in range(n_steps):
        t = step·dt
        self._k_detach(t); self._k_advect(t); self._k_collide(); self._k_remove()
        if dump_every and step % dump_every == 0: self._dump(step)
    self._k_count()
    return scores (B,)
```

커널 4개를 한 커널로 합치면 빠르지만 디버깅이 어렵다. 먼저 분리해서 맞추고 성능 단계에서 합친다.

`evaluate`는 `batch_evaluate([(pose, nozzle)])[0]`에 부위별 값을 붙여 `EvalResult`로 감싼다.

## 단계 10. 후보 격리 테스트 (필수)

`tests/test_particles.py`:
- **동일 입력 격리**: 후보 2개에 같은 자세 → 두 점수 완전 일치.
- **단독 vs 배치 일치**: 후보 A, B를 각각 `B=1`로 평가한 점수 == `B=2`로 함께 평가한 점수. 이것이 깨지면 인덱스가 섞인 것.
- **결정론**: 같은 시드 두 번 → 일치.
- **세기 0**: 모든 노즐 `strength=0` → 제거 0.
- **제트 일치**: 단계 4 검증.
- **질량 보존**: `state` 0/1/2 개수 합 == `B·N` 매 스텝.

개발 중 설정: `N=1000, B=2, duration 0.5 s`.

## 단계 11. 성능 (3주차)

목표: `N=20000, B=100, 1500 스텝`을 RTX 4090에서 200 ms 이하, Vulkan 노트북 GPU에서 2 s 이하.

- 커널 4개 → 1개 병합.
- 캡슐 루프: `K_max`를 후보별 실제 `n_caps`로 제한.
- `jet_velocity`를 이탈 판정과 적분에서 두 번 부르지 않게 한 번 계산 후 공유.
- 부유 입자 비율이 낮으면 `state` 분기로 대부분 스레드가 일찍 끝나므로 SoA 유지.
- `ti.profiler`로 커널별 시간 확인.

## 단계 12. 프레임 덤프 (1시간)

`dump_every > 0`이면 `outputs/<exp_id>/frames/<step>.npz`에 `pos, state, part_init, candidate` 저장 (`docs/interfaces.md` 형식). E 트랙이 애니메이션에 쓴다. 최적화 중에는 0.

## 완료 기준

- [ ] 환경 기록에 동작 백엔드가 적혀 있다.
- [ ] 원통 1개 + 노즐 1개에서 입자가 이탈하고 날아가고 부스 밖으로 나가는 것이 `plot_body`류 산점도나 GGUI로 보인다.
- [ ] 격리 테스트 6개 통과.
- [ ] B·D 병합 후: `run_optimize.py --evaluator particle --scenario default --max-evals 500`이 돈다.
- [ ] E2: 무작위 자세 500개에서 패치판과 순위 상관 0.8 이상 (D의 `compare_evaluators.py`).
- [ ] 성능 목표 달성 또는 미달 시 병목 커널과 수치를 기록.

## 밀릴 때

3주차 말까지 격리 테스트와 E2를 못 넘으면 C에 알려 패치판으로 데이터셋 생성을 시작하게 한다. 입자판은 완성 후 E4 상위 후보 재평가에만 쓴다.

## 환경 기록

| 머신 | 백엔드 | SAXPY 1e7 | 비고 |
|---|---|---|---|
| Windows 11 노트북, RTX 5060 Laptop (8 GB, driver 592.82), Ryzen AI 9 | **cuda** (채택) | **0.494 ms** | Python 3.13.5 / taichi 1.7.4 / numpy 2.5.1. `ti.init(arch=ti.cuda)` 성공 |
| 같은 머신 | vulkan | 0.661 ms | 폴백 후보. 정상 동작 |
| 같은 머신 | cpu (x64) | 5.119 ms | 참고용. CI 용도로만 |

측정 방법: `1e7` 원소 `y = a*x + y` 커널. JIT 워밍업 1회 후 20회 평균, 각 백엔드를 **별도 프로세스**에서 측정 (`ti.init`은 프로세스당 1회).
`ti.init`이 요청한 arch로 뜨지 않고 조용히 폴백하는 경우가 있어 `ti.lang.impl.current_cfg().arch`로 실제 백엔드를 확인했다.

- 개발 기본 백엔드는 **cuda**. 0.5 ms < 1 ms 기준이므로 GPU가 정상적으로 잡혔다.
- `ParticleEvaluator(arch=...)`는 `"gpu"`를 받으면 cuda → vulkan → cpu 순으로 시도한다.
- 다른 세션과 GPU를 공유하므로 개발 중 설정은 `N=1000, B=2, duration 0.5 s`로 유지한다 (`00_common.md` 7절).
