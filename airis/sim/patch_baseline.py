"""패치 기반 빠른 평가기. 소유자: D.

역할
- 입자판 완성 전까지 최적화 파이프라인을 끝까지 돌리는 기준선
- 입자판 디버깅 시 비교 대상
- 밀리초 단위여야 함 (numpy 벡터화)

흐름 (`docs/tracks/D_patch_baseline.md` 단계 4)

    state   = build_body(body, pose, scenario)
    u_mn    = velocity_field_per_nozzle(patch_pos + d·n, nozzle, t=0, cfg,
                                        surface_normals=patch_normal)        # (M,N,3), 4.2b 보정
    visible = occlusion(state, nozzle, cfg)                                  # (M,N)
    u       = sum_m u_mn · visible[m]                                        # (N,3)
    tau     = scoring.wall_shear(u, patch_normal, cfg)                       # (N,)
    R       = scoring.removal_fraction(tau, cfg)                             # (N,)
    R_part  = 면적 가중 평균 per part
    score, disc = scoring.score(R_part, pose, scenario, cfg)

`build_body` 직후 부스 밖 자세 불가 규칙(`00_common.md` 5절)을 먼저 검사하고, 걸리면 제트
계산 없이 벽을 넘은 거리에 비례한 벌점 `score = -1 - 10·d_out`을 돌려준다.

물리 상수는 전부 `physics_cfg`(= `configs/physics.yaml`)에서 읽는다.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import scoring
from .body import build_body as _default_build_body
from .interface import Evaluator
from .jet import velocity_field_per_nozzle as _default_velocity_field_per_nozzle
from .scenario import load_nozzle_layout
from .types import PART_NAMES, BodyParams, BodyState, EvalResult, NozzleConfig, PoseParams, Scenario

_EPS = 1e-12
# 후보 거르기 여유. float32 반올림보다 충분히 커서 거르기가 항상 보수적이 되게 한다.
_CONE_SLACK = 1e-4
# 부스 밖 자세의 벌점 (00_common.md 5절): score = INFEASIBLE_BASE - INFEASIBLE_SLOPE_PER_M · d_out.
INFEASIBLE_BASE = -1.0
INFEASIBLE_SLOPE_PER_M = 10.0


def _segment_segment_dist_sq(s1: np.ndarray, d1: np.ndarray,
                             p2: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """선분 (s1, s1+d1)과 (p2, p2+d2) 사이 최단 거리 제곱. 입력 (Q,3), 출력 (Q,).

    Ericson, *Real-Time Collision Detection* 5.1.9 (ClosestPtSegmentSegment)를
    1-D 배치로 벡터화했다. 길이 0인 선분(머리 구 캡슐 등)도 처리한다.
    """
    r = s1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    a_safe = np.maximum(a, _EPS)
    e_safe = np.maximum(e, _EPS)

    # 일반 경우. 평행이면 denom = 0 -> s = 0에서 시작.
    denom = a * e - b * b
    ok = denom > _EPS
    s = np.where(ok, np.clip((b * f - c * e) / np.where(ok, denom, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / e_safe
    # t가 [0,1] 밖이면 끝점으로 고정하고 s를 다시 구한다.
    s = np.where(t < 0.0, np.clip(-c / a_safe, 0.0, 1.0),
                 np.where(t > 1.0, np.clip((b - c) / a_safe, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)

    # 퇴화 경우는 일반식이 풀리지 않으므로 덮어쓴다 (f = d2·r 이 정확히 0이 되어
    # 위의 t 재계산 분기에 들어가지 않는다).
    point_capsule = e <= _EPS          # 캡슐 축이 점: t = 0, s는 점에 대한 투영
    s = np.where(point_capsule, np.clip(-c / a_safe, 0.0, 1.0), s)
    t = np.where(point_capsule, 0.0, t)
    point_ray = a <= _EPS              # 선분이 점: s = 0, t는 캡슐 축에 대한 투영
    s = np.where(point_ray, 0.0, s)
    t = np.where(point_ray, np.clip(f / e_safe, 0.0, 1.0), t)

    w = r + s[:, None] * d1 - t[:, None] * d2
    return np.einsum("ij,ij->i", w, w)


def occlusion(state: BodyState, nozzle: NozzleConfig, physics_cfg: dict) -> np.ndarray:
    """(M, N) bool. 노즐 m이 패치 n을 볼 수 있으면 True.

    `docs/tracks/D_patch_baseline.md` 단계 3.

    1. 뒷면: `normal · (nozzle_pos - patch_pos) <= 0` 이면 노즐이 패치 뒤에 있으므로
       거리 계산 없이 가림.
    2. 후보 거르기 (보수적, float32): 노즐에서 본 캡슐 경계 구의 원뿔 밖에 있거나,
       경계 구가 패치보다 멀리 있으면 그 캡슐은 이 광선을 가릴 수 없다. 노즐마다
       행렬곱 한 번으로 (N, K)를 판정한다. 여유를 두어 남기기만 하고 버리지는 않는다.
    3. 정확 판정 (float64, 후보만): 선분(패치 표면 `patch_pos + d·normal` -> 노즐)과
       캡슐 축의 최단 거리가 반지름보다 작으면 가림. 자기 캡슐은 제외한다.

    2단계는 결과를 바꾸지 않고 계산량만 줄인다 (전수 판정과 결과가 같음을 테스트로 확인).
    """
    delta = float(physics_cfg["air"]["wall_offset_m"])

    pos = np.asarray(state.patch_pos, dtype=np.float64)          # (N,3)
    normal = np.asarray(state.patch_normal, dtype=np.float64)    # (N,3)
    npos = np.asarray(nozzle.positions, dtype=np.float64)        # (M,3)
    caps = np.asarray(state.capsules, dtype=np.float64)          # (K,7)

    # 1. 뒷면 노즐.
    facing = npos @ normal.T - np.einsum("ij,ij->i", normal, pos)[None, :] > 0.0   # (M,N)
    visible = facing.copy()
    if caps.shape[0] == 0 or not facing.any():
        return visible

    start = pos + delta * normal                                 # (N,3) 선분 시작점
    p2 = caps[:, 0:3]                                            # (K,3) 캡슐 축 시작
    d2 = caps[:, 3:6] - p2                                       # (K,3) 캡슐 축 방향
    radius = caps[:, 6]                                          # (K,)
    center = p2 + 0.5 * d2
    bound = 0.5 * np.sqrt(np.einsum("ij,ij->i", d2, d2)) + radius   # 경계 구 반지름
    own = (None if state.patch_capsule is None
           else np.asarray(state.patch_capsule, dtype=np.int64))
    start32 = start.astype(np.float32)

    # 2. 노즐마다 원뿔·깊이로 후보 (m, n, k)를 거른다.
    cand_m, cand_n, cand_k = [], [], []
    for m in range(npos.shape[0]):
        n_sel = np.flatnonzero(facing[m])
        if n_sel.size == 0:
            continue
        ray = start32[n_sel] - npos[m].astype(np.float32)                    # (Nm,3)
        ray_len = np.sqrt(np.einsum("ij,ij->i", ray, ray))                   # (Nm,)

        to_center = center - npos[m]                                         # (K,3)
        center_dist = np.sqrt(np.einsum("ij,ij->i", to_center, to_center))   # (K,)
        nozzle_inside = center_dist <= bound       # 원뿔이 정의되지 않음 -> 항상 후보
        sin_half = np.where(nozzle_inside, 1.0, bound / np.maximum(center_dist, _EPS))
        cos_half = np.sqrt(np.maximum(0.0, 1.0 - sin_half * sin_half))
        axis = to_center / np.maximum(center_dist, _EPS)[:, None]

        in_cone = (ray @ axis.T.astype(np.float32)
                   >= ray_len[:, None] * (cos_half - _CONE_SLACK).astype(np.float32)[None, :])
        in_reach = ((center_dist - bound).astype(np.float32)[None, :]
                    <= ray_len[:, None] + np.float32(_CONE_SLACK))
        cand = (in_cone & in_reach) | nozzle_inside[None, :]
        if own is not None:
            cand[np.arange(n_sel.size), own[n_sel]] = False

        qi, kk = np.nonzero(cand)
        cand_m.append(np.full(qi.size, m))
        cand_n.append(n_sel[qi])
        cand_k.append(kk)

    cm = np.concatenate(cand_m)
    if cm.size == 0:
        return visible
    cn = np.concatenate(cand_n)
    ck = np.concatenate(cand_k)

    # 3. 후보만 정확히 판정한다.
    s1 = start[cn]
    dist_sq = _segment_segment_dist_sq(s1, npos[cm] - s1, p2[ck], d2[ck])
    hit = dist_sq < radius[ck] ** 2
    visible[cm[hit], cn[hit]] = False
    return visible


def _area_weighted_by_part(values: np.ndarray, area: np.ndarray,
                           part: np.ndarray) -> np.ndarray:
    """부위별 면적 가중 평균 -> (len(PART_NAMES),). 패치가 없는 부위는 0."""
    n_parts = len(PART_NAMES)
    weighted = np.bincount(part, weights=values * area, minlength=n_parts)[:n_parts]
    total = np.bincount(part, weights=area, minlength=n_parts)[:n_parts]
    return np.divide(weighted, total, out=np.zeros(n_parts), where=total > 0.0)


def booth_excess(patch_pos: np.ndarray, booth: dict) -> float:
    """벽을 넘은 거리 d_out (m). 부스 안이면 0.

    `d_out = max(0, max_i(|y_i| - width/2), max_i(z_i - height))` (`00_common.md` 5절).
    x 방향은 열린 문이라 검사하지 않는다. 경계와 같은 값은 안쪽이다.
    """
    pos = np.asarray(patch_pos, dtype=np.float64)
    if pos.size == 0:
        return 0.0
    over_wall = np.abs(pos[:, 1]).max() - 0.5 * float(booth["width_m"])
    over_ceiling = pos[:, 2].max() - float(booth["height_m"])
    return float(max(0.0, over_wall, over_ceiling))


def outside_booth(patch_pos: np.ndarray, booth: dict) -> bool:
    """패치가 하나라도 옆벽(|y| > width/2)이나 천장(z > height) 밖이면 True."""
    return booth_excess(patch_pos, booth) > 0.0


def infeasible_score(d_out: float) -> float:
    """부스 밖 자세의 점수 `-1 - 10·d_out`. 조금 닿으면 -1에 가깝고 많이 나갈수록 낮다."""
    return INFEASIBLE_BASE - INFEASIBLE_SLOPE_PER_M * float(d_out)


class PatchEvaluator(Evaluator):
    """입자 없이 패치별 벽면 전단만으로 제거율을 구하는 numpy 평가기.

    `build_body`와 `velocity_field_per_nozzle`는 B 소유다. 기본값으로 B의 구현을
    쓰되, B 병합 전이나 단위 테스트에서는 `tests/fakes.py`의 가짜를 주입할 수 있게
    생성자 인자로 뺐다 (`docs/tracks/00_common.md` 3절).

    - `patches_per_m2`: `build_body`에 넘기는 패치 밀도. `None`이면 `build_body` 기본값
      (2000, 검증용). 최적화 루프는 400을 쓴다 (D 문서 완료 기준).
    - `booth`: 부스 밖 판정에 쓰는 `{"width_m", "height_m", ...}`. `None`이면
      `configs/nozzles.yaml`의 `booth`.
    """

    def __init__(self, physics_cfg: dict,
                 build_body: Callable[..., BodyState] | None = None,
                 velocity_field_per_nozzle: Callable[..., np.ndarray] | None = None,
                 *, patches_per_m2: float | None = None,
                 booth: dict | None = None):
        self.cfg = physics_cfg
        self._build_body = build_body or _default_build_body
        self._velocity_field_per_nozzle = (
            velocity_field_per_nozzle or _default_velocity_field_per_nozzle)
        self.patches_per_m2 = patches_per_m2
        self.booth = booth if booth is not None else load_nozzle_layout()["booth"]

    def build_state(self, body: BodyParams, pose: PoseParams, scenario: Scenario) -> BodyState:
        """`evaluate`가 쓰는 것과 같은 밀도로 몸을 만든다."""
        if self.patches_per_m2 is None:
            # 밀도 인자를 받지 않는 주입 함수(가짜 몸)도 쓸 수 있게 기본값일 때는 넘기지 않는다.
            return self._build_body(body, pose, scenario)
        return self._build_body(body, pose, scenario, patches_per_m2=self.patches_per_m2)

    def evaluate(self, pose: PoseParams, nozzle: NozzleConfig,
                 body: BodyParams, scenario: Scenario) -> EvalResult:
        state = self.build_state(body, pose, scenario)
        d_out = booth_excess(state.patch_pos, self.booth)
        if d_out > 0.0:
            return EvalResult(
                score=infeasible_score(d_out),
                removal_by_part=np.zeros(len(PART_NAMES)),
                total_removal=0.0,
                discomfort=scoring.discomfort(pose, scenario),
                extra={"infeasible": True, "d_out": d_out},
            )

        delta = float(self.cfg["air"]["wall_offset_m"])

        normal = np.asarray(state.patch_normal, dtype=np.float64)
        area = np.asarray(state.patch_area, dtype=np.float64)
        part = np.asarray(state.patch_part, dtype=np.int64)

        # 공기 속도는 표면에서 wall_offset_m 만큼 띄운 곳에서 조회한다 (4.2).
        probe = np.asarray(state.patch_pos, dtype=np.float64) + delta * normal

        # 정상 상태 평가라 t = 0. 펄스는 무시한다 (00_common.md 4.1).
        # 법선을 넘겨 충돌 제트 -> 벽면 제트 보정(00_common.md 4.2b)을 받는다.
        u_mn = np.asarray(self._velocity_field_per_nozzle(
            probe, nozzle, 0.0, self.cfg, surface_normals=normal))                       # (M,N,3)
        visible = occlusion(state, nozzle, self.cfg)                                      # (M,N)
        # 가려진 노즐의 기여를 빼고 합산한다. (M,N,3) 마스크 곱 대신 축약 합으로 한 번에.
        u = np.einsum("mnk,mn->nk", u_mn, visible.astype(u_mn.dtype)).astype(np.float64)

        tau = scoring.wall_shear(u, normal, self.cfg)
        removal = scoring.removal_fraction(tau, self.cfg)

        removal_by_part = _area_weighted_by_part(removal, area, part)
        total_removal = float((removal * area).sum() / area.sum()) if area.sum() > 0 else 0.0
        total, disc = scoring.score(removal_by_part, pose, scenario, self.cfg)

        return EvalResult(
            score=total,
            removal_by_part=removal_by_part,
            total_removal=total_removal,
            discomfort=disc,
            extra={"tau": tau, "removal": removal, "visible_frac": float(visible.mean()),
                   "infeasible": False},
        )
