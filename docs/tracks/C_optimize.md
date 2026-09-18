# 트랙 C: 최적화·실험 운영

**목표**: 시나리오 제약 안에서 자세 벡터를 CMA-ES로 탐색하는 루프, 실험 로그, 실행 스크립트를 만든다. 이후 민감도 실험(E3), 기준선 비교(E4), 데이터셋 생성(E5)의 실행 인프라가 된다.

**파일**: `airis/optimize/encoding.py`, `cmaes_runner.py`, `explog.py`, `dataset.py`, `scripts/run_optimize.py`, `run_baselines.py`, `run_e4.py`, `tests/test_encoding.py`, `tests/test_cmaes_dummy.py`

**의존성**: `Evaluator` 인터페이스만. 평가기는 D의 `PatchEvaluator`가 병합될 때까지 `DummyEvaluator`로 대체. 노즐은 B 병합 전까지 `tests/fakes.py`의 `fake_nozzles`.

---

## 단계 1. 인코딩 (`encoding.py`, 2시간)

```python
class PoseEncoder:
    def __init__(self, scenario: Scenario): ...
    free_keys: list[str]          # PoseParams 필드 중 scenario.fixed_pose에 없는 것, 필드 순서 유지
    dim: int                      # len(free_keys)
    def encode(self, pose: PoseParams) -> np.ndarray      # [-1,1]^dim
    def decode(self, x: np.ndarray) -> PoseParams         # clip 후 역변환, fixed_pose 적용
    def default_x(self) -> np.ndarray                     # PoseParams() 기본값의 인코딩
```

정규화: `x = 2·(θ − lo)/(hi − lo) − 1`, `(lo, hi) = scenario.pose_bounds[key]`. `pose_bounds`에 없는 키는 `default` 시나리오 값으로 채운다 (`load_scenarios`가 상속을 이미 처리하므로 실제로는 항상 있음).

`decode`는 `x`를 [-1, 1]로 clip한 뒤 변환한다. CMA-ES가 경계 밖을 제안해도 안전하다.

## 단계 2. 더미 평가기 (`tests/fakes.py`에 추가 요청 또는 `optimize/dummy.py`, 1시간)

D 소유 파일을 건드리지 않기 위해 `airis/optimize/dummy.py`에 둔다.

```python
class DummyEvaluator(Evaluator):
    def __init__(self, target: PoseParams, scenario: Scenario): ...
    def evaluate(self, pose, nozzle, body, scenario) -> EvalResult:
        # score = -||encode(pose) - encode(target)||²  (정규화 공간 거리)
```

CMA-ES가 `target`을 찾아내는지로 루프를 검증한다.

## 단계 3. CMA-ES 루프 (`cmaes_runner.py`, 반나절)

```python
@dataclass
class OptResult:
    best_pose: PoseParams
    best_score: float
    best_result: EvalResult
    history: list[dict]           # per generation: gen, evals, best, mean, sigma
    n_evals: int
    exp_id: str

def run_cmaes(evaluator, body, scenario, nozzle, *, max_evals=3000, seed=0,
              popsize=100, sigma0=0.5, tol_stagnation_gens=30, log_dir=None) -> OptResult
```

구현:

```python
enc = PoseEncoder(scenario)
es = cma.CMAEvolutionStrategy(enc.default_x(), sigma0,
        {"bounds": [-1, 1], "popsize": popsize, "seed": seed,
         "maxfevals": max_evals, "verbose": -9})
while not es.stop():
    X = es.ask()                                        # list of (dim,)
    poses = [enc.decode(x) for x in X]
    scores = evaluator.batch_evaluate([(p, nozzle) for p in poses], body, scenario)
    es.tell(X, (-scores).tolist())                      # cma는 최소화
    # 세대 로그, best 갱신, 정체 판정
```

- 정체: 최근 `tol_stagnation_gens` 세대 동안 best 개선이 `1e-4` 미만이면 종료.
- `best_result`는 best 자세를 `evaluator.evaluate`로 한 번 더 평가해 부위별 값을 채운다.
- 시드가 같으면 결과가 완전히 같아야 한다 (평가기가 결정론적이라는 전제).

## 단계 4. 실험 로그 (`explog.py`, 2시간)

```python
def new_exp_id(prefix: str) -> str            # "{prefix}_{YYYYmmdd_HHMMSS}_{6자리 해시}"
def git_commit() -> str                        # subprocess rev-parse --short HEAD, 실패 시 "nogit"
def file_hash(path) -> str                     # sha256 앞 8자리
def write_run(log_dir, exp_id, *, scenario, body, nozzle_hash, physics_hash,
              commit, seed, args, result: OptResult)
```

저장 구조:

```
outputs/<exp_id>/
  meta.json        # scenario, body, hashes, commit, seed, args, n_evals, best_score
  history.csv      # gen, evals, best, mean, sigma
  best.json        # best_pose (dict), removal_by_part, discomfort
outputs/index.csv  # 한 줄씩 append: exp_id, 날짜, scenario, best_score, commit
```

`docs/experiments.md`는 사람이 쓰는 요약이고, `index.csv`가 기계용 원장이다.

## 단계 5. 실행 스크립트 (`scripts/run_optimize.py`, 2시간)

```
python scripts/run_optimize.py --scenario default --evaluator patch --max-evals 3000 --seed 0
  --evaluator {dummy,patch,particle}
  --body '{"height_m":1.6}'      # 기본값 덮어쓰기, 생략 시 BodyParams()
  --popsize 100 --sigma0 0.5
  --tag e4                       # exp_id 접두어
```

- `patch`: `PatchEvaluator(load_physics())`. D 병합 전에는 ImportError 안내 후 종료.
- `particle`: `ParticleEvaluator(load_physics())`. A 병합 전 동일.
- 노즐: `load_nozzles()`가 `NotImplementedError`면 `fake_nozzles()`로 폴백하고 meta에 `nozzle_source: fake` 기록.
- 종료 시 best 자세, 점수, 부위별 제거율, 소요 시간을 표로 출력.

## 단계 6. 기준선 평가 (`scripts/run_baselines.py`, 1시간)

E4의 B0, B1을 평가한다.

| 조건 | PoseParams | 비고 |
|---|---|---|
| B0 | `PoseParams()` 기본값 | 정면, 팔 내림 |
| B1 | `PoseParams()`에 `torso_yaw = 0, 30, …, 330` (12개)를 각각 평가한 평균 | 퓨리움 공식 안내 "서서 몸을 회전". `score`·`removal_by_part`·`total_removal`은 가능한 yaw의 평균, `discomfort`는 기본 자세 값(0). 전부 불가면 불가 |
| B2 | `shoulder_abduction=180, elbow_flexion=0`, 나머지 기본 | 일반 에어샤워 안내 "만세". 체형에 따라 천장(2.15 m)에 걸리면 불가로 기록 |

("팔 수평 90°"는 기준 장비에서 벽 밖이라 제외.) 세 시나리오 × 세 조건을 평가해 `outputs/baselines_<날짜>.csv`에 저장. 휠체어는 `fixed_pose` 적용. B1의 yaw는 시나리오 `pose_bounds` 안의 값만 평균한다 (휠체어는 [−45, 45]라 −30/0/30 — 휠체어는 깊이 0.886 m 게이트 안에서 제자리 회전이 불가능하다). `run_e4.py`의 개선율도 B0·B1·B2 세 기준으로 낸다.

## 단계 7. E4 러너 (`scripts/run_e4.py`, 2주차, 2시간)

시나리오 3개 × 시드 5개를 순차 실행하고 요약한다.

- 각 실행은 `run_cmaes` 호출, `tag=e4`.
- 요약 표: 시나리오별 best_score 평균·표준편차, B0·B1 대비 개선율, 시드 간 best_pose 표준편차(수렴 확인).
- `outputs/e4_summary.csv`, 자세 비교용 `best_poses.json`.

## 단계 8. 민감도 러너 (`scripts/run_e3.py`, 3주차)

`--param adhesion.critical_shear_pa_median --factors 0.5 0.75 1 1.5 2`처럼 physics 설정 하나를 바꿔 `run_e4` 축소판(시드 2개)을 돌린다. 설정 덮어쓰기는 `load_physics()` 결과 dict를 수정해 평가기에 넘긴다 (`configs/` 파일은 건드리지 않음). 결과: 인자값별 best_pose와 상위 10개 후보의 순위 상관.

## 단계 9. 데이터셋 생성 (`dataset.py`, 3~4주차)

체형 샘플링:

| 변수 | 분포 |
|---|---|
| `height_m` | N(1.68, 0.09), clip [1.45, 1.95] |
| `shoulder_width_m` | `0.245·height + N(0, 0.015)` |
| `torso_depth_m` | `0.13·height + N(0, 0.015)` |
| `arm_length_m` | `0.365·height + N(0, 0.02)` |
| `leg_length_m` | `0.50·height + N(0, 0.025)` |

- 시드 고정 샘플러 `sample_bodies(n, seed) -> list[BodyParams]`.
- 체형 × 시나리오마다 `run_cmaes` (max_evals 2000, popsize 50).
- 결과를 `docs/interfaces.md` 스키마대로 parquet 한 행. 100행마다 flush.
- 재개: 시작 시 기존 parquet의 `(body_idx, scenario)` 키를 읽어 건너뛴다.
- 병렬: `multiprocessing.Pool`로 프로세스당 평가기 하나. 입자판은 GPU 하나에 프로세스 1~2개까지.

## 단계 10. 테스트

`tests/test_encoding.py`:
- encode → decode 왕복이 clip 범위 안에서 원본과 1e-6 이내.
- 휠체어 시나리오: `dim == 5`, decode 결과의 `hip_flexion == 90`.
- 범위 밖 `x`가 clip되어 bounds 안 값이 나옴.

`tests/test_cmaes_dummy.py`:
- 목표 자세를 `DummyEvaluator`에 주고 `max_evals=1500, popsize=20`으로 실행 → 정규화 공간 거리 0.05 이내.
- 같은 seed 두 번 → `history` 완전 일치.

## 완료 기준

- [ ] `run_optimize.py --evaluator dummy`가 끝까지 돌고 `outputs/<exp_id>/`에 세 파일이 생긴다.
- [ ] 더미 수렴 테스트와 인코딩 테스트 통과.
- [ ] D 병합 후 `--evaluator patch --scenario wheelchair`가 돌고, 결과 자세가 `hip_flexion=90`을 유지하며 팔 각도만 바뀐다.
- [ ] `run_baselines.py`가 세 시나리오의 B0, B1 점수를 출력한다.
- [ ] 1주차 체크포인트: 패치판(`--patches-per-m2 400`)으로 default 시나리오 3,000회 평가가 1분 안에 끝난다. `run_optimize.py --patches-per-m2`는 `PatchEvaluator(patches_per_m2=...)`로 전달한다.

## 병합 후 알림

병합되면 A 세션에 "`run_optimize.py --evaluator particle`이 `ParticleEvaluator.batch_evaluate`를 호출하니 그 시그니처에 맞추라"고 전달한다.
