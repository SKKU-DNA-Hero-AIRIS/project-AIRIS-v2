"""계획 모델(F)의 PlanSpace 와 최적화(C)의 PlanEncoder 규격 대조. 소유자: F.

docs/interfaces.md "계획 모델", docs/tracks/00_common.md 4.7. 키 순서·단계 시간 식은 C 의
tests/test_plan_encoding.py 가 이미 고정한다. 여기서는 그 밖의 규격을 본다.

- 같은 설정(configs/physics.yaml)에서 읽은 범위가 같다 (1단계 yaw 는 0~90° 로 접은 범위)
- 단계 수 K 가 달라도 단계 시간 식이 같다
- normalize·clip_plan 을 거친 계획은 PlanSpace 범위 안이고, 원래 단위 벡터(from_plan = raw_vector)가 같다
- PlanSpace.to_plan 의 출력에 clip_plan 을 다시 걸어도 쾌적 상한·풍량 한도 말고는 바뀌지 않는다 (투영이 겹치지 않는다)

torch 없이 돈다 (flow.py 는 torch 를 함수 안에서만 import 한다).
"""
import numpy as np
import pytest

from airis.model.flow import POSE_KEYS, PlanSpace
from airis.optimize.plan_encoding import PlanEncoder, PlanLimits
from airis.sim import ZONE_NAMES, Phase, Plan, PoseParams
from airis.sim.scenario import load_physics, load_scenarios

SCENARIOS = ("default", "pregnant", "wheelchair")


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios()


@pytest.fixture(scope="module")
def cfg():
    return load_physics()


def _random_plan(rng, scenario, n_phases, duration_bounds=(5.0, 20.0), wide=False):
    """시나리오 범위 안(wide=True 면 범위 밖까지)의 무작위 계획."""
    phases = []
    for _ in range(n_phases):
        vals = {}
        for k in POSE_KEYS:
            if k in scenario.fixed_pose:
                vals[k] = float(scenario.fixed_pose[k])
                continue
            lo, hi = scenario.pose_bounds[k]
            pad = 0.3 * (hi - lo) if wide else 0.0
            vals[k] = float(rng.uniform(lo - pad, hi + pad))
        phases.append(vals)
    lo_t, hi_t = duration_bounds
    total = rng.uniform(lo_t - (3.0 if wide else 0.0), hi_t + (10.0 if wide else 0.0))
    weights = rng.dirichlet(np.ones(n_phases))
    times = [max(0.1, total * w) for w in weights]
    zones = rng.uniform(-0.2 if wide else 0.0, 1.5 if wide else 1.0, len(ZONE_NAMES))
    return Plan([Phase(PoseParams(**v), t) for v, t in zip(phases, times)], zones)


def spec_bounds(enc: PlanEncoder, key: str) -> tuple[float, float]:
    """docs/plan_extension.md 2절·00_common 4.7 의 범위를 PlanEncoder 의 공개 값(limits, pose_encoder)으로 적는다.
    1단계 yaw 만 대칭 정규화로 [0, min(90, max|범위|)] 에 든다."""
    if key == "duration_s":
        return enc.limits.duration_bounds_s
    if key.startswith("share_"):
        return 0.0, 1.0
    if key.startswith("zone_"):
        return 0.0, enc.limits.s_max
    phase, _, pose_key = key.partition("_")
    lo, hi = enc.pose_encoder.bounds_for(pose_key)
    if phase == "p1" and pose_key == "torso_yaw":
        return 0.0, min(90.0, max(abs(lo), abs(hi)))
    return lo, hi


@pytest.mark.parametrize("name", SCENARIOS)
def test_bounds_match_encoder_from_same_config(scenarios, cfg, name):
    """시나리오 하나로 만든 PlanSpace 는 같은 설정의 PlanEncoder 공개 값(limits, pose_encoder)으로 적은 규격 범위와 같다."""
    sc = scenarios[name]
    space = PlanSpace.from_config([sc], cfg)
    enc = PlanEncoder(sc, PlanLimits.from_config(cfg))
    assert space.keys == enc.keys
    assert space.n_phases == enc.limits.n_phases and space.min_phase_s == enc.limits.min_phase_s
    bounds = space.to_dict()
    for key in enc.keys:
        _, _, pose_key = key.partition("_")
        if key.startswith("p") and pose_key in sc.fixed_pose:
            continue                       # 고정 변수: 마스크로 빠지므로 PlanSpace 는 폭만 채운다
        assert bounds[key] == pytest.approx(list(spec_bounds(enc, key))), key


def test_union_space_covers_every_scenario_encoder(scenarios, cfg):
    """여러 시나리오로 학습한 PlanSpace 범위는 시나리오마다의 PlanEncoder 범위를 모두 덮는다."""
    space = PlanSpace.from_config([scenarios[n] for n in SCENARIOS], cfg)
    bounds = space.to_dict()
    for name in SCENARIOS:
        sc = scenarios[name]
        enc = PlanEncoder(sc, PlanLimits.from_config(cfg))
        for key in enc.keys:
            _, _, pose_key = key.partition("_")
            if key.startswith("p") and pose_key in sc.fixed_pose:
                continue
            lo, hi = spec_bounds(enc, key)
            assert bounds[key][0] <= lo + 1e-9 and hi <= bounds[key][1] + 1e-9, (name, key)


@pytest.mark.parametrize("n_phases, duration_bounds", [(2, (5.0, 20.0)), (3, (6.0, 20.0)), (1, (5.0, 20.0))])
def test_durations_match_for_any_phase_count(scenarios, n_phases, duration_bounds):
    sc = scenarios["default"]
    space = PlanSpace.from_scenarios([sc], n_phases=n_phases, duration_bounds_s=duration_bounds)
    enc = PlanEncoder(sc, PlanLimits(n_phases=n_phases, duration_bounds_s=duration_bounds))
    assert space.keys == enc.keys
    rng = np.random.default_rng(0)
    for _ in range(50):
        total = rng.uniform(*duration_bounds)
        shares = list(rng.uniform(-0.2, 1.2, n_phases - 1))
        assert space.durations(total, shares) == pytest.approx(enc.durations(total, shares))


@pytest.mark.parametrize("name", SCENARIOS)
def test_normalized_clipped_plans_lie_in_space(scenarios, cfg, name):
    """C 의 normalize → clip_plan 을 거친 계획(= 계획 데이터셋의 행)은 PlanSpace 범위 안이고,
    PlanSpace.from_plan 과 PlanEncoder.raw_vector 가 같은 벡터를 낸다. 1단계 yaw 는 0~90°."""
    sc = scenarios[name]
    space = PlanSpace.from_config([scenarios[n] for n in SCENARIOS], cfg)
    enc = PlanEncoder(sc, PlanLimits.from_config(cfg))
    lo = np.array([space.to_dict()[k][0] for k in space.keys])
    hi = np.array([space.to_dict()[k][1] for k in space.keys])
    free = space.mask(sc) > 0              # 고정 변수(휠체어 hip·knee 90°)는 마스크로 빠지고 to_plan 이 다시 채운다
    rng = np.random.default_rng(1)
    for _ in range(200):
        plan = enc.clip_plan(enc.normalize(_random_plan(rng, sc, enc.limits.n_phases)))
        raw = space.from_plan(plan)
        assert np.allclose(raw, enc.raw_vector(plan))
        inside = (raw >= lo - 1e-9) & (raw <= hi + 1e-9)
        assert np.all(inside[free]), [k for k, ok, f in zip(space.keys, inside, free) if f and not ok]
        assert 0.0 <= plan.phases[0].pose.torso_yaw <= 90.0


@pytest.mark.parametrize("name", SCENARIOS)
def test_encode_decode_round_trip_through_space(scenarios, cfg, name):
    """데이터셋 행 → PlanSpace 정규화 → to_plan 이 원래 계획으로 돌아온다 (학습 입력과 샘플 출력의 규격이 같다)."""
    sc = scenarios[name]
    space = PlanSpace.from_config([scenarios[n] for n in SCENARIOS], cfg)
    enc = PlanEncoder(sc, PlanLimits.from_config(cfg))
    rng = np.random.default_rng(2)
    for _ in range(50):
        plan = enc.clip_plan(enc.normalize(_random_plan(rng, sc, enc.limits.n_phases)))
        back = space.to_plan(space.encode(space.from_plan(plan))[0], sc)
        assert np.allclose(space.from_plan(back), space.from_plan(plan), atol=1e-6)


@pytest.mark.parametrize("name", SCENARIOS)
def test_clip_after_to_plan_only_applies_caps(scenarios, cfg, name):
    """to_plan 이 이미 자세·시간·세기 범위를 지키므로, clip_plan 이 바꾸는 것은 쾌적 상한·풍량 한도뿐이다.
    clip_plan 은 멱등이다."""
    from airis.sim.scenario import zone_strength_caps

    sc = scenarios[name]
    space = PlanSpace.from_config([scenarios[n] for n in SCENARIOS], cfg)
    enc = PlanEncoder(sc, PlanLimits.from_config(cfg))
    caps = np.minimum(zone_strength_caps(sc), enc.limits.s_max)
    rng = np.random.default_rng(3)
    for x in rng.uniform(-1.3, 1.3, size=(200, space.dim)):
        plan = space.to_plan(x, sc)
        clipped = enc.clip_plan(plan)
        for a, b in zip(plan.phases, clipped.phases):
            assert np.allclose(a.pose.to_vector(), b.pose.to_vector())
            assert a.duration_s == pytest.approx(b.duration_s)
        if enc.limits.cap_ratio >= 1.0:          # 풍량 한도가 걸리지 않으면 세기는 쾌적 상한만 바뀐다
            assert np.allclose(clipped.zone_strengths, np.minimum(plan.zone_strengths, caps))
        again = enc.clip_plan(clipped)
        assert np.allclose(space.from_plan(again), space.from_plan(clipped))

