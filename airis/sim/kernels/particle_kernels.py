"""입자 시뮬레이터의 Taichi 부분. 소유자: A.

`airis/sim/particles.py`(호스트 오케스트레이션)와 분리한 이유는 디버깅이다.
필드 할당과 커널 4개(이탈 / 적분 / 충돌 / 제거)가 여기 모여 있고, 호스트는
numpy 준비와 업로드만 한다.

수식 출처는 전부 `docs/tracks/00_common.md`다.
- 4.1 자유 제트 속도장  -> `ParticleFields.jet_velocity` (ti.func)
- 4.2 벽면 전단         -> `k_detach`
- 4.3 이탈 판정         -> `k_detach` (입자별 tau_crit 비교)
- 4.5 항력              -> `k_advect`

배치 규약 (`A_particles.md` 단계 2)
- 후보 B개 × 입자 N개를 길이 B·N으로 평탄화. 입자 i의 후보 인덱스는 `i // N`.
- 커널은 `n_act * N`까지만 순회하므로 비활성 슬롯은 건드리지 않는다.
- 입자 간 상호작용이 없고 스텝 중 atomic이 없으므로, 어떤 후보를 어느 슬롯에
  넣든 산술이 동일하다 = "단독 vs 배치 일치"가 성립한다 (단계 10).
"""
# `from __future__ import annotations`를 쓰면 안 된다: Taichi 커널은 인자 주석을
# 실제 타입 객체로 읽는다.
import typing

import numpy as np
import taichi as ti

# ---------------------------------------------------------------- 상수 인덱스
# 물리 상수를 f32 필드 하나에 담아 올린다. 파이썬 스칼라 속성을 커널 안에서 읽으면
# 컴파일 시점에 상수로 굳어 값을 바꿔도 반영되지 않는다 (Taichi의 알려진 함정).
C_JET_D = 0          # 노즐 지름 D
C_JET_U0 = 1         # 출구 속도 (strength 곱하기 전)
C_JET_K = 2          # 감쇠 상수 K
C_JET_SPREAD = 3     # halfwidth_spread_rate
C_PULSE_ON = 4       # 0/1
C_PULSE_PERIOD = 5
C_PULSE_DUTY = 6
C_RHO_AIR = 7
C_MU = 8             # 공기 점성
C_CF = 9             # 벽면 마찰 계수
C_WALL_OFF = 10      # air.wall_offset_m
C_RHO_P = 11         # 입자 밀도
C_P_REDEP = 12       # 재부착 확률
C_BOOTH_L = 13
C_BOOTH_W = 14
C_BOOTH_H = 15
C_GZ = 16            # 중력 z 성분 (-9.81)
NUM_CONST = 17

# 부위 인덱스 (types.PART_NAMES 순서). 커널 안에서 컴파일 상수로 쓴다.
PART_TORSO_FRONT = 1
PART_TORSO_BACK = 2

SQRT2LN2 = 1.177     # 반속도 반경 -> 가우시안 sigma 환산 (00_common.md 4.1)
GRAVITY_Z = -9.81

# ------------------------------------------------------------------ ti.init
_ARCH_FALLBACK = {
    "gpu": ("cuda", "vulkan", "cpu"),
    "cuda": ("cuda",),
    "vulkan": ("vulkan",),
    "cpu": ("cpu",),
}
_ARCH_OBJ = {"cuda": ti.cuda, "vulkan": ti.vulkan, "cpu": ti.cpu}

_initialized_arch: typing.Optional[str] = None


def init_taichi(arch: str = "gpu", **kwargs) -> str:
    """프로세스당 한 번만 `ti.init`. 이미 초기화되어 있으면 건너뛴다 (단계 1).

    두 번째 `ti.init`은 기존 SNode 트리를 날려 먼저 만든 평가기의 필드를
    무효화하므로 절대 다시 호출하지 않는다. 실제로 잡힌 백엔드 이름을 돌려준다.
    """
    global _initialized_arch
    if _initialized_arch is not None:
        return _initialized_arch

    order = _ARCH_FALLBACK.get(arch)
    if order is None:
        raise ValueError("알 수 없는 arch: {!r} (gpu/cuda/vulkan/cpu)".format(arch))

    errors = []
    for name in order:
        try:
            ti.init(arch=_ARCH_OBJ[name], default_fp=ti.f32, default_ip=ti.i32, **kwargs)
        except Exception as exc:                  # noqa: BLE001 - 백엔드별 예외가 제각각
            errors.append("{}: {}: {}".format(name, type(exc).__name__, exc))
            continue
        # ti.init은 요청한 arch를 못 쓰면 조용히 cpu로 폴백한다. 실제 값을 확인한다.
        actual = ti.lang.impl.current_cfg().arch
        if actual != _ARCH_OBJ[name] and name != "cpu":
            errors.append("{}: 조용한 폴백 -> {}".format(name, actual))
            continue
        _initialized_arch = str(actual).replace("Arch.", "")
        return _initialized_arch

    raise RuntimeError("사용 가능한 Taichi 백엔드가 없다:\n  " + "\n  ".join(errors))


def current_arch() -> typing.Optional[str]:
    return _initialized_arch


# ------------------------------------------------------------------- 필드/커널
@ti.data_oriented
class ParticleFields:
    """단계 2의 데이터 레이아웃. `__init__`에서 고정 할당하고 재사용한다."""

    def __init__(self, max_candidates: int, particles_per_candidate: int,
                 max_capsules: int = 64, max_nozzles: int = 64):
        self.B = int(max_candidates)
        self.N = int(particles_per_candidate)
        self.K = int(max_capsules)
        self.M = int(max_nozzles)
        bn = self.B * self.N

        # FieldsBuilder로 한 트리에 모아 둔다. 암묵적 root를 쓰면 평가기를 여러 개
        # 만들 때 SNode 트리가 32개 한도를 넘을 수 있고 destroy()도 못 한다.
        fb = ti.FieldsBuilder()

        self.pos = ti.Vector.field(3, ti.f32)
        self.vel = ti.Vector.field(3, ti.f32)
        self.normal = ti.Vector.field(3, ti.f32)
        self.diam = ti.field(ti.f32)
        self.tau_crit = ti.field(ti.f32)
        self.tau_crit_respawn = ti.field(ti.f32)   # 재부착 시 다시 쓸 임계 전단 (단계 5)
        self.rand_redep = ti.field(ti.f32)         # 재부착 판정용 난수 (단계 7)
        self.state = ti.field(ti.i32)              # 0 부착, 1 부유, 2 제거
        self.part = ti.field(ti.i32)               # 현재 부위 (재부착으로 바뀔 수 있음)
        self.part_init = ti.field(ti.i32)          # 초기 부위. 집계는 이쪽으로 (단계 7)
        fb.dense(ti.i, bn).place(
            self.pos, self.vel, self.normal, self.diam, self.tau_crit,
            self.tau_crit_respawn, self.rand_redep,
            self.state, self.part, self.part_init,
        )

        self.capsules = ti.field(ti.f32)           # (B, K, 7) = [p0(3), p1(3), r]
        fb.dense(ti.ijk, (self.B, self.K, 7)).place(self.capsules)
        self.cap_part = ti.field(ti.i32)           # -1 = 가림 전용
        fb.dense(ti.ij, (self.B, self.K)).place(self.cap_part)
        self.n_caps = ti.field(ti.i32)
        # 후보별 몸 전방 단위 벡터. BodyState.capsule_part는 캡슐당 하나라 몸통 캡슐이
        # torso_front로만 들어오므로, 재부착 부위의 앞/뒤는 충돌 법선·전방 부호로 가른다.
        self.body_fwd = ti.Vector.field(3, ti.f32)
        fb.dense(ti.i, self.B).place(self.n_caps, self.body_fwd)

        self.noz_pos = ti.Vector.field(3, ti.f32)
        self.noz_dir = ti.Vector.field(3, ti.f32)
        self.noz_strength = ti.field(ti.f32)
        self.noz_phase = ti.field(ti.f32)
        fb.dense(ti.i, self.M).place(self.noz_pos, self.noz_dir,
                                     self.noz_strength, self.noz_phase)

        self.count_init = ti.field(ti.i32)
        self.count_removed = ti.field(ti.i32)
        fb.dense(ti.ij, (self.B, 5)).place(self.count_init, self.count_removed)

        self.cst = ti.field(ti.f32)
        fb.dense(ti.i, NUM_CONST).place(self.cst)
        self.state_hist = ti.field(ti.i32)         # 질량 보존 테스트용 (단계 10)
        fb.dense(ti.i, 3).place(self.state_hist)

        self.n_noz = ti.field(ti.i32)
        fb.place(self.n_noz)

        self._tree = fb.finalize()

    def destroy(self) -> None:
        """SNode 트리 해제. 평가기를 많이 만드는 테스트에서 호출한다."""
        if self._tree is not None:
            self._tree.destroy()
            self._tree = None

    # ------------------------------------------------------ 4.1 자유 제트 속도장
    @ti.func
    def jet_velocity(self, p, t):
        """00_common.md 4.1을 그대로. 노즐 M개 기여의 벡터 합."""
        u = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        big_d = self.cst[C_JET_D]
        k_dec = self.cst[C_JET_K]
        spread = self.cst[C_JET_SPREAD]
        l_core = k_dec * big_d
        for m in range(self.n_noz[None]):
            d = self.noz_dir[m]
            r = p - self.noz_pos[m]
            s = r.dot(d)
            if s > 0.0:                                   # s <= 0 이면 기여 없음
                rho = (r - s * d).norm()
                u0 = self.cst[C_JET_U0] * self.noz_strength[m]
                u_c = u0
                if s > l_core:
                    u_c = u0 * l_core / s                 # 코어 밖에서 1/s 감쇠
                sig = 0.5 * big_d / SQRT2LN2 + spread * s / SQRT2LN2
                gate = 1.0
                if self.cst[C_PULSE_ON] > 0.5:
                    ph = t / self.cst[C_PULSE_PERIOD] + self.noz_phase[m]
                    frac = ph - ti.floor(ph)
                    if frac >= self.cst[C_PULSE_DUTY]:
                        gate = 0.0
                u += (u_c * ti.exp(-rho * rho / (2.0 * sig * sig)) * gate) * d
        return u

    # ----------------------------------------------------------- 단계 5. 이탈
    @ti.kernel
    def k_detach(self, t: ti.f32, n_act: ti.i32):
        for i in range(n_act * self.N):
            if self.state[i] == 0:
                nrm = self.normal[i]
                u = self.jet_velocity(self.pos[i] + self.cst[C_WALL_OFF] * nrm, t)
                u_t = u - u.dot(nrm) * nrm                          # 4.2 접선 성분
                tau = 0.5 * self.cst[C_RHO_AIR] * self.cst[C_CF] * u_t.norm_sqr()
                if tau > self.tau_crit[i]:
                    self.state[i] = 1
                    self.vel[i] = u_t * 0.1 + nrm * 0.05            # 작은 초기 속도

    # -------------------------------------------------- 단계 6. 부유 입자 적분
    @ti.kernel
    def k_advect(self, t: ti.f32, dt: ti.f32, n_act: ti.i32):
        for i in range(n_act * self.N):
            if self.state[i] == 1:
                u = self.jet_velocity(self.pos[i], t)
                v = self.vel[i]
                d = self.diam[i]
                re = self.cst[C_RHO_AIR] * (u - v).norm() * d / self.cst[C_MU]
                # Schiller-Naumann. 4.5는 Re < 1000 구간만 적으므로 그 위는
                # 표준 뉴턴 영역(Cd = 0.44 -> f = Cd·Re/24)으로 잇는다.
                f = 1.0 + 0.15 * ti.pow(ti.max(re, 1e-12), 0.687)
                if re >= 1000.0:
                    f = 0.0183 * re
                tau_p = self.cst[C_RHO_P] * d * d / (18.0 * self.cst[C_MU] * f)
                v_new = (u + (v - u) * ti.exp(-dt / tau_p)
                         + ti.Vector([0.0, 0.0, self.cst[C_GZ]], dt=ti.f32) * dt)
                self.vel[i] = v_new
                self.pos[i] = self.pos[i] + v_new * dt

    # ------------------------------------------- 단계 7. 캡슐 충돌과 재부착
    @ti.kernel
    def k_collide(self, n_act: ti.i32):
        for i in range(n_act * self.N):
            if self.state[i] == 1:
                b = i // self.N
                p = self.pos[i]
                v = self.vel[i]
                for k in range(self.n_caps[b]):
                    p0 = ti.Vector([self.capsules[b, k, 0], self.capsules[b, k, 1],
                                    self.capsules[b, k, 2]], dt=ti.f32)
                    p1 = ti.Vector([self.capsules[b, k, 3], self.capsules[b, k, 4],
                                    self.capsules[b, k, 5]], dt=ti.f32)
                    rad = self.capsules[b, k, 6]
                    ab = p1 - p0
                    denom = ab.dot(ab)
                    seg = 0.0
                    if denom > 1e-12:
                        seg = ti.min(ti.max((p - p0).dot(ab) / denom, 0.0), 1.0)
                    c = p0 + seg * ab                               # 축 선분 위 최근접점
                    dv = p - c
                    dist = dv.norm()
                    if dist < rad:
                        n_hit = ti.Vector([0.0, 0.0, 1.0], dt=ti.f32)
                        if dist > 1e-9:
                            n_hit = dv / dist
                        p = c + n_hit * (rad + 1e-4)                # 표면 밖으로
                        cp = self.cap_part[b, k]
                        if cp == PART_TORSO_FRONT and n_hit.dot(self.body_fwd[b]) <= 0.0:
                            cp = PART_TORSO_BACK
                        if cp >= 0 and self.rand_redep[i] < self.cst[C_P_REDEP]:
                            self.state[i] = 0                       # 몸에 재부착
                            self.part[i] = cp
                            self.normal[i] = n_hit
                            self.tau_crit[i] = self.tau_crit_respawn[i]
                            v = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                            break
                        v_n = v.dot(n_hit) * n_hit                  # 반사
                        v = (v - v_n) * 0.8 - v_n * 0.3
                self.pos[i] = p
                self.vel[i] = v

    # ------------------------------------------------- 단계 8. 부스 이탈 = 제거
    @ti.kernel
    def k_remove(self, n_act: ti.i32):
        for i in range(n_act * self.N):
            if self.state[i] == 1:
                p = self.pos[i]
                if (p[0] < 0.0 or p[0] > self.cst[C_BOOTH_L]
                        or ti.abs(p[1]) > 0.5 * self.cst[C_BOOTH_W]
                        or p[2] < 0.0 or p[2] > self.cst[C_BOOTH_H]):
                    self.state[i] = 2

    @ti.kernel
    def k_count(self, n_act: ti.i32):
        """제거된 입자를 **초기** 부위로 집계 (재부착으로 part가 바뀌므로)."""
        for i in range(n_act * self.N):
            if self.state[i] == 2:
                ti.atomic_add(self.count_removed[i // self.N, self.part_init[i]], 1)

    @ti.kernel
    def k_state_hist(self, n_act: ti.i32):
        for i in range(n_act * self.N):
            ti.atomic_add(self.state_hist[self.state[i]], 1)

    @ti.kernel
    def k_zero_removed(self, n_act: ti.i32):
        for b, q in ti.ndrange(n_act, 5):
            self.count_removed[b, q] = 0

    @ti.kernel
    def k_zero_hist(self):
        for q in range(3):
            self.state_hist[q] = 0

    # ------------------------------------------------- 검증용 속도장 조회 커널
    @ti.kernel
    def k_probe_velocity(self, pts: ti.types.ndarray(), out: ti.types.ndarray(),
                         n_pts: ti.i32, t: ti.f32):
        """단계 4 검증용. 임의의 점에서 jet_velocity를 그대로 뽑는다."""
        for i in range(n_pts):
            u = self.jet_velocity(
                ti.Vector([pts[i, 0], pts[i, 1], pts[i, 2]], dt=ti.f32), t)
            out[i, 0] = u[0]
            out[i, 1] = u[1]
            out[i, 2] = u[2]


def pack_constants(cfg: dict, booth: dict) -> np.ndarray:
    """physics.yaml + nozzles.yaml booth -> 커널 상수 배열."""
    jet, air = cfg["jet"], cfg["air"]
    pulse = jet.get("pulse") or {}
    c = np.zeros(NUM_CONST, dtype=np.float32)
    c[C_JET_D] = jet["nozzle_diameter_m"]
    c[C_JET_U0] = jet["exit_velocity_mps"]
    c[C_JET_K] = jet["decay_constant"]
    c[C_JET_SPREAD] = jet["halfwidth_spread_rate"]
    c[C_PULSE_ON] = 1.0 if pulse.get("enabled", False) else 0.0
    c[C_PULSE_PERIOD] = pulse.get("period_s") or 1.0
    c[C_PULSE_DUTY] = pulse.get("duty", 1.0)
    c[C_RHO_AIR] = air["density"]
    c[C_MU] = air["viscosity"]
    c[C_CF] = air["friction_coeff"]
    c[C_WALL_OFF] = air["wall_offset_m"]
    c[C_RHO_P] = cfg["particles"]["density_kg_m3"]
    c[C_P_REDEP] = cfg["adhesion"]["redeposition_prob"]
    c[C_BOOTH_L] = booth["length_m"]
    c[C_BOOTH_W] = booth["width_m"]
    c[C_BOOTH_H] = booth["height_m"]
    c[C_GZ] = GRAVITY_Z
    return c
