"""입자 기반 평가기 (Taichi GPU). 소유자: A.

설계 원칙
- 처음부터 배치 구조: 후보 B개 × 입자 N개를 길이 B·N 필드로 평탄화 (`kernels/`)
- 난수는 전부 호스트 numpy에서 만들어 필드에 복사한다. GPU 난수는 백엔드마다 달라
  결정론이 깨진다 (`docs/tracks/A_particles.md` 단계 3)
- 스텝: 이탈 판정 -> 부유 입자 적분 -> 캡슐 충돌/재부착 -> 부스 이탈 제거

구현 상태 (A_particles.md 기준)
- 단계 1~10 완료. 단계 11(성능: 커널 병합), 12(프레임 덤프)는 미구현.
- `batch_evaluate`/`evaluate`는 B의 `build_body`를 호출한다. B 병합 전에는
  `batch_evaluate_states`에 BodyState를 직접 넘겨 쓴다 (테스트가 이 경로).
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from .body import build_body
from .interface import Evaluator
from .kernels import ParticleFields, init_taichi, pack_constants
from .kernels.particle_kernels import PART_TORSO_BACK, PART_TORSO_FRONT
from .scenario import load_nozzle_layout
from .types import (
    PART_NAMES, BodyParams, BodyState, EvalResult, NozzleConfig, PoseParams, Scenario,
)

N_PARTS = len(PART_NAMES)
assert PART_NAMES[PART_TORSO_FRONT] == "torso_front" and PART_NAMES[PART_TORSO_BACK] == "torso_back"
SURFACE_LIFT_M = 1e-4          # 입자를 패치 표면에서 띄우는 거리 (단계 3)


class ParticleEvaluator(Evaluator):
    """후보 B개를 한 번에 평가하는 입자 평가기.

    `max_candidates`, `particles_per_candidate`는 필드 고정 할당 크기다. 매 호출
    재할당은 느리므로 한 번 만들고 재사용한다. 그보다 적은 후보는 앞쪽 슬롯만 쓴다.
    """

    def __init__(self, physics_cfg: dict, arch: str = "gpu", *,
                 max_candidates: int = 100,
                 particles_per_candidate: int | None = None,
                 max_capsules: int = 64, max_nozzles: int = 64,
                 booth: dict | None = None,
                 duration_s: float | None = None):
        self.cfg = physics_cfg
        self.arch = init_taichi(arch)          # 프로세스당 1회. 이미 되어 있으면 건너뜀

        sim = physics_cfg["simulation"]
        self.N = int(particles_per_candidate
                     or physics_cfg["particles"]["count_per_candidate"])
        self.max_candidates = int(max_candidates)
        self.dt = float(sim["dt_s"])
        self.duration_s = float(duration_s if duration_s is not None else sim["duration_s"])
        self.seed = int(sim["seed"])
        self.booth = booth if booth is not None else load_nozzle_layout()["booth"]

        self.f = ParticleFields(self.max_candidates, self.N, max_capsules, max_nozzles)
        self.f.cst.from_numpy(pack_constants(physics_cfg, self.booth))

        # 호스트 스테이징 버퍼. from_numpy는 필드 전체 형상을 요구하므로 최대 크기로 둔다.
        bn = self.max_candidates * self.N
        self._h = {
            "pos": np.zeros((bn, 3), np.float32),
            "vel": np.zeros((bn, 3), np.float32),
            "normal": np.zeros((bn, 3), np.float32),
            "diam": np.zeros(bn, np.float32),
            "tau_crit": np.zeros(bn, np.float32),
            "tau_crit_respawn": np.zeros(bn, np.float32),
            "rand_redep": np.zeros(bn, np.float32),
            "state": np.zeros(bn, np.int32),
            "part": np.zeros(bn, np.int32),
            "part_init": np.zeros(bn, np.int32),
        }
        self._h_caps = np.zeros((self.max_candidates, self.f.K, 7), np.float32)
        self._h_cap_part = np.full((self.max_candidates, self.f.K), -1, np.int32)
        self._h_n_caps = np.zeros(self.max_candidates, np.int32)
        self._h_body_fwd = np.zeros((self.max_candidates, 3), np.float32)
        self._h_count_init = np.zeros((self.max_candidates, N_PARTS), np.int32)

    def destroy(self) -> None:
        self.f.destroy()

    # ------------------------------------------------------------------ 공개 API
    def evaluate(self, pose: PoseParams, nozzle: NozzleConfig,
                 body: BodyParams, scenario: Scenario) -> EvalResult:
        state = build_body(body, pose, scenario)
        return self.batch_evaluate_states([state], [pose], nozzle, scenario)[0]

    def batch_evaluate(self, candidates: Sequence[tuple[PoseParams, NozzleConfig]],
                       body: BodyParams, scenario: Scenario) -> np.ndarray:
        if not candidates:
            return np.zeros(0, dtype=np.float32)
        nozzle = _shared_nozzle(candidates)
        poses = [pose for pose, _ in candidates]
        states = [build_body(body, pose, scenario) for pose in poses]   # 호스트, ~10 ms × B
        results = self.batch_evaluate_states(states, poses, nozzle, scenario)
        return np.array([r.score for r in results], dtype=np.float32)

    def batch_evaluate_states(self, states: Sequence[BodyState],
                              poses: Sequence[PoseParams],
                              nozzle: NozzleConfig, scenario: Scenario,
                              *, step_callback=None) -> list[EvalResult]:
        """BodyState를 직접 받는 배치 평가. `batch_evaluate`의 본체.

        `step_callback(step, evaluator, n_act)`는 매 스텝 뒤 호출된다 (테스트/디버그용).
        """
        n_act = len(states)
        if n_act != len(poses):
            raise ValueError("states와 poses 길이가 다르다")
        if not 0 < n_act <= self.max_candidates:
            raise ValueError(f"후보 수 {n_act}가 범위 (1..{self.max_candidates}) 밖")

        self._init_candidates(states, poses, nozzle, n_act)
        self._upload_nozzles(nozzle)

        n_steps = int(round(self.duration_s / self.dt))
        f = self.f
        for step in range(n_steps):
            t = step * self.dt
            f.k_detach(t, n_act)
            f.k_advect(t, self.dt, n_act)
            f.k_collide(n_act)
            f.k_remove(n_act)
            if step_callback is not None:
                step_callback(step, self, n_act)
        f.k_count(n_act)

        removed = f.count_removed.to_numpy()[:n_act]
        init = self._h_count_init[:n_act]
        results = []
        for b in range(n_act):
            removal_by_part = (removed[b] / np.maximum(init[b], 1)).astype(np.float32)
            total = float(removed[b].sum()) / self.N
            discomfort = _discomfort(poses[b], scenario)
            score = _score(removal_by_part, discomfort, self.cfg)
            results.append(EvalResult(
                score=score,
                removal_by_part=removal_by_part,
                total_removal=total,
                discomfort=discomfort,
                extra={
                    "evaluator": "particle",
                    "count_init": init[b].copy(),
                    "count_removed": removed[b].copy(),
                    "n_steps": n_steps,
                    "seed": self._candidate_seed(poses[b], nozzle),
                },
            ))
        return results

    # ---------------------------------------------------------- 디버그 조회
    def state_counts(self, n_act: int) -> np.ndarray:
        """state 0/1/2 개수 (3,). 질량 보존 확인용."""
        self.f.k_zero_hist()
        self.f.k_state_hist(n_act)
        return self.f.state_hist.to_numpy()

    def snapshot(self, b: int) -> dict[str, np.ndarray]:
        """후보 b의 현재 입자 상태를 호스트로 복사 (시각화용, 느림)."""
        sl = slice(b * self.N, (b + 1) * self.N)
        return {
            "pos": self.f.pos.to_numpy()[sl],
            "state": self.f.state.to_numpy()[sl],
            "part_init": self.f.part_init.to_numpy()[sl],
        }

    def probe_velocity(self, points: np.ndarray, nozzle: NozzleConfig,
                       t: float = 0.0) -> np.ndarray:
        """Taichi `jet_velocity`를 임의의 점 (P,3)에서 평가 -> (P,3). 단계 4 검증용."""
        self._upload_nozzles(nozzle)
        pts = np.ascontiguousarray(points, dtype=np.float32)
        out = np.zeros_like(pts)
        self.f.k_probe_velocity(pts, out, pts.shape[0], float(t))
        return out

    # ------------------------------------------------------ 단계 3. 호스트 초기화
    def _candidate_seed(self, pose: PoseParams, nozzle: NozzleConfig) -> int:
        """후보 **내용**으로 정한 시드.

        문서의 `seed + b`(슬롯 인덱스)를 그대로 쓰면 같은 후보도 몇 번째 슬롯에
        들어가느냐에 따라 난수가 달라져 단계 10의 "단독 vs 배치 일치"가 원리적으로
        깨진다. 슬롯 대신 (자세, 노즐)의 해시를 더해 "후보마다 다른 시드, 같은 입력
        같은 결과"를 슬롯과 무관하게 지킨다.
        """
        h = hashlib.blake2b(digest_size=8)
        h.update(np.asarray(pose.to_vector(), dtype=np.float32).tobytes())
        h.update(np.asarray(nozzle.positions, dtype=np.float32).tobytes())
        h.update(np.asarray(nozzle.directions, dtype=np.float32).tobytes())
        h.update(np.asarray(nozzle.strengths, dtype=np.float32).tobytes())
        return (self.seed + int.from_bytes(h.digest(), "little")) % (2 ** 63)

    def _init_candidates(self, states, poses, nozzle, n_act: int) -> None:
        self._h_n_caps[:] = 0
        self._h_cap_part[:] = -1
        self._h_caps[:] = 0.0
        self._h_count_init[:] = 0
        for b in range(n_act):
            rng = np.random.default_rng(self._candidate_seed(poses[b], nozzle))
            self._init_candidate(b, states[b], poses[b], rng)

        f = self.f
        for name, arr in self._h.items():
            getattr(f, name).from_numpy(arr)
        f.capsules.from_numpy(self._h_caps)
        f.cap_part.from_numpy(self._h_cap_part)
        f.n_caps.from_numpy(self._h_n_caps)
        f.body_fwd.from_numpy(self._h_body_fwd)
        f.count_init.from_numpy(self._h_count_init)
        f.k_zero_removed(n_act)

    def _init_candidate(self, b: int, state: BodyState, pose: PoseParams,
                        rng: np.random.Generator) -> None:
        """A_particles.md 단계 3을 그대로. 난수 호출 순서가 결정론의 일부다."""
        n = self.N
        sl = slice(b * n, (b + 1) * n)
        cfg = self.cfg

        # 1. 면적 비례 배분
        area = np.asarray(state.patch_area, dtype=np.float64)
        if area.size == 0 or area.sum() <= 0:
            raise ValueError(f"후보 {b}: 패치 면적 합이 0")
        cum = np.cumsum(area)
        cum /= cum[-1]
        idx = np.searchsorted(cum, rng.random(n), side="right")
        idx = np.minimum(idx, area.size - 1)

        # 2. 위치 / 법선 / 부위
        normal = np.asarray(state.patch_normal, dtype=np.float32)[idx]
        self._h["pos"][sl] = np.asarray(state.patch_pos, dtype=np.float32)[idx] \
            + SURFACE_LIFT_M * normal
        self._h["normal"][sl] = normal
        part = np.asarray(state.patch_part, dtype=np.int32)[idx]
        self._h["part"][sl] = part
        self._h["part_init"][sl] = part

        # 3. 지름, 임계 전단 (로그 정규). respawn과 재부착 난수도 여기서 미리 뽑는다.
        size = cfg["particles"]["size_distribution_um"]
        adh = cfg["adhesion"]
        tau_med = adh["critical_shear_pa_median"] * adh["fabric_roughness_factor"]
        self._h["diam"][sl] = rng.lognormal(np.log(size["median"] * 1e-6),
                                            size["sigma_log"], n)
        self._h["tau_crit"][sl] = rng.lognormal(np.log(tau_med),
                                                adh["critical_shear_sigma_log"], n)
        self._h["tau_crit_respawn"][sl] = rng.lognormal(np.log(tau_med),
                                                        adh["critical_shear_sigma_log"], n)
        self._h["rand_redep"][sl] = rng.random(n)

        # 4. 상태
        self._h["state"][sl] = 0
        self._h["vel"][sl] = 0.0

        # 5. 캡슐과 부위 집계
        caps = np.asarray(state.capsules, dtype=np.float32).reshape(-1, 7)
        k = caps.shape[0]
        if k > self.f.K:
            raise ValueError(f"후보 {b}: 캡슐 {k}개 > max_capsules {self.f.K}")
        cap_part = (np.asarray(state.capsule_part, dtype=np.int32)
                    if state.capsule_part is not None else np.full(k, -1, np.int32))
        self._h_caps[b, :k] = caps
        self._h_cap_part[b, :k] = cap_part
        self._h_n_caps[b] = k
        self._h_body_fwd[b] = body_forward(pose)
        self._h_count_init[b] = np.bincount(part, minlength=N_PARTS)[:N_PARTS]

    def _upload_nozzles(self, nozzle: NozzleConfig) -> None:
        m = nozzle.count
        if m > self.f.M:
            raise ValueError(f"노즐 {m}개 > max_nozzles {self.f.M}")
        pos = np.zeros((self.f.M, 3), np.float32)
        dirs = np.zeros((self.f.M, 3), np.float32)
        strength = np.zeros(self.f.M, np.float32)
        phase = np.zeros(self.f.M, np.float32)
        d = np.asarray(nozzle.directions, dtype=np.float64)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)       # 4.1은 단위 방향을 가정
        pos[:m] = nozzle.positions
        dirs[:m] = d
        strength[:m] = nozzle.strengths
        if nozzle.pulse_phase is not None:
            phase[:m] = nozzle.pulse_phase
        self.f.noz_pos.from_numpy(pos)
        self.f.noz_dir.from_numpy(dirs)
        self.f.noz_strength.from_numpy(strength)
        self.f.noz_phase.from_numpy(phase)
        self.f.n_noz[None] = m


def body_forward(pose: PoseParams) -> np.ndarray:
    """몸 전방 단위 벡터 (cos yaw, sin yaw, 0). B의 build_body가 torso_yaw를 z축
    반시계 회전으로 적용하므로 같은 규약이다. torso_pitch 기울기는 무시하는 근사라
    앞/뒤 경계 근처(법선이 거의 좌우 또는 위아래인 곳)에서만 판정이 갈릴 수 있다."""
    yaw = np.radians(pose.torso_yaw)
    return np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)


# ---------------------------------------------------------------- 점수 (4.4)
# TODO(A): D 병합 후 `airis/sim/scoring.py`의 구현으로 교체한다. 지금은
# `docs/tracks/00_common.md` 4.4를 그대로 인라인 구현했다. 부위별 제거율은 입자 수 가중.
def _discomfort(pose: PoseParams, scenario: Scenario) -> float:
    default = PoseParams()
    total = 0.0
    for k, c_k in scenario.discomfort_weights.items():
        bounds = scenario.pose_bounds.get(k)
        if not c_k or bounds is None:
            continue
        lo, hi = bounds
        if hi <= lo:
            continue
        total += c_k * abs(getattr(pose, k) - getattr(default, k)) / (hi - lo)
    return float(total)


def _score(removal_by_part: np.ndarray, discomfort: float, cfg: dict) -> float:
    sc = cfg["scoring"]
    weights = sc["part_weights"]
    gain = sum(weights.get(name, 0.0) * float(removal_by_part[i])
               for i, name in enumerate(PART_NAMES))
    return float(gain - sc["discomfort_weight"] * discomfort)


def _shared_nozzle(candidates: Sequence[tuple[PoseParams, NozzleConfig]]) -> NozzleConfig:
    """노즐은 후보 공통 (고정 입력). 다른 노즐이 섞여 있으면 조용히 틀리지 않게 막는다."""
    first = candidates[0][1]
    for _, nz in candidates[1:]:
        if nz is first:
            continue
        if not (np.array_equal(nz.positions, first.positions)
                and np.array_equal(nz.directions, first.directions)
                and np.array_equal(nz.strengths, first.strengths)):
            raise ValueError("입자판은 후보 공통 노즐만 지원한다 (노즐은 고정 입력)")
    return first
