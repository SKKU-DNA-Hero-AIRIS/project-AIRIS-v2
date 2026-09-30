"""PoseEncoder 단위 테스트. 소유자: C. docs/tracks/C_optimize.md 단계 10."""
import numpy as np
import pytest

from airis.optimize.encoding import POSE_FIELDS, PoseEncoder
from airis.sim import PoseParams
from airis.sim.scenario import load_scenarios


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios()


def _random_pose_in_bounds(enc: PoseEncoder, rng: np.random.Generator) -> PoseParams:
    """시나리오 제약 안의 임의 자세."""
    return enc.decode(rng.uniform(-1.0, 1.0, size=enc.dim))


@pytest.mark.parametrize("name", ["default", "pregnant", "wheelchair"])
def test_encode_decode_roundtrip(scenarios, name):
    enc = PoseEncoder(scenarios[name])
    rng = np.random.default_rng(0)
    for _ in range(50):
        pose = _random_pose_in_bounds(enc, rng)
        back = enc.decode(enc.encode(pose))
        for key in POSE_FIELDS:
            assert getattr(back, key) == pytest.approx(getattr(pose, key), abs=1e-6)


def test_default_scenario_dim_is_seven(scenarios):
    enc = PoseEncoder(scenarios["default"])
    assert enc.dim == 7
    assert enc.free_keys == POSE_FIELDS


def test_wheelchair_dim_and_fixed_pose(scenarios):
    enc = PoseEncoder(scenarios["wheelchair"])
    assert enc.dim == 5
    assert "hip_flexion" not in enc.free_keys
    assert "knee_flexion" not in enc.free_keys

    rng = np.random.default_rng(1)
    for _ in range(10):
        pose = enc.decode(rng.uniform(-1.0, 1.0, size=enc.dim))
        assert pose.hip_flexion == 90
        assert pose.knee_flexion == 90


@pytest.mark.parametrize("name", ["default", "pregnant", "wheelchair"])
def test_out_of_range_x_is_clipped(scenarios, name):
    scenario = scenarios[name]
    enc = PoseEncoder(scenario)
    high = enc.decode(np.full(enc.dim, 5.0))
    low = enc.decode(np.full(enc.dim, -5.0))
    for key in enc.free_keys:
        lo, hi = enc.bounds_for(key)
        assert getattr(high, key) == pytest.approx(hi)
        assert getattr(low, key) == pytest.approx(lo)
        assert lo <= getattr(high, key) <= hi
        assert lo <= getattr(low, key) <= hi


@pytest.mark.parametrize("name", ["default", "pregnant", "wheelchair"])
def test_encode_stays_in_unit_box(scenarios, name):
    enc = PoseEncoder(scenarios[name])
    # 범위를 한참 벗어난 자세도 [-1, 1] 로 clip 된다.
    wild = PoseParams(**{k: 1e4 for k in POSE_FIELDS})
    x = enc.encode(wild)
    assert x.shape == (enc.dim,)
    assert np.all(x >= -1.0) and np.all(x <= 1.0)


def test_default_x_matches_default_pose(scenarios):
    enc = PoseEncoder(scenarios["default"])
    back = enc.decode(enc.default_x())
    base = PoseParams()
    for key in POSE_FIELDS:
        assert getattr(back, key) == pytest.approx(getattr(base, key), abs=1e-6)


@pytest.mark.parametrize("name", ["default", "pregnant"])
def test_all_free_keys_bounded_by_config(scenarios, name):
    # B #10 병합 후 knee_flexion 범위도 configs/scenarios.yaml 에서 온다 (코드 대체 표 없음).
    scenario = scenarios[name]
    enc = PoseEncoder(scenario)
    assert set(enc.free_keys) <= set(scenario.pose_bounds)
    assert enc.bounds_for("knee_flexion") == (0.0, 30.0)


# ---------- 실행 시점 범위 좁히기 (--pose-bound) ----------

def test_narrow_scenario_intersects_and_keeps_original(scenarios):
    """좁힌 범위는 원래 범위와 교집합이고, 원본 Scenario 는 그대로다."""
    from airis.optimize import cli

    base = scenarios["default"]
    before = dict(base.pose_bounds)
    narrowed = cli.narrow_scenario(base, ["shoulder_abduction=0,90", "shoulder_flexion=-30,90"])

    assert narrowed.pose_bounds["shoulder_abduction"] == (0.0, 90.0)
    assert narrowed.pose_bounds["shoulder_flexion"] == (-30.0, 90.0)
    assert narrowed.pose_bounds["torso_yaw"] == before["torso_yaw"]   # 안 건드린 변수는 그대로
    assert base.pose_bounds == before, "원본 시나리오를 바꾸면 안 된다"
    assert cli.narrow_scenario(base, []) is base

    # 넓히려 해도 원래 범위를 못 넘는다 (교집합). 조용히 줄이지 않고 경고를 낸다.
    with pytest.warns(RuntimeWarning, match="교집합"):
        wide = cli.narrow_scenario(base, ["shoulder_abduction=-90,900"])
    assert wide.pose_bounds["shoulder_abduction"] == before["shoulder_abduction"]

    # 좁힌 범위는 PoseEncoder 에 그대로 들어간다.
    assert PoseEncoder(narrowed).bounds_for("shoulder_abduction") == (0.0, 90.0)
    decoded = PoseEncoder(narrowed).decode(np.ones(PoseEncoder(narrowed).dim))
    assert decoded.shoulder_abduction == pytest.approx(90.0)


def test_narrow_scenario_rejects_bad_specs(scenarios):
    from airis.optimize import cli

    base = scenarios["default"]
    bad = [
        "shoulder_abduction=0",          # 쉼표 없음
        "shoulder_abduction",            # = 없음
        "shoulder_abduction=a,b",        # 수가 아님
        "shoulder_abduction=nan,90",     # nan
        "shoulder_abduction=90,0",       # lo > hi 를 조용히 뒤집지 않는다
        "shoulder_abduction=90,90",      # 한 점
        "없는변수=0,90",
        "shoulder_abduction=200,300",    # 시나리오 범위 밖
    ]
    for spec in bad:
        with pytest.raises(SystemExit):
            cli.narrow_scenario(base, [spec])

    seated = scenarios["wheelchair"]
    fixed = next(iter(seated.fixed_pose), None)
    if fixed:                                  # 고정된 변수는 좁히지 못한다
        with pytest.raises(SystemExit):
            cli.narrow_scenario(seated, [f"{fixed}=0,10"])


def test_starts_in_bounds_skips_out_of_range(scenarios):
    """좁힌 상자 밖 시작점은 조용히 잘리지 않고 걸러진다 (통합 2026-09-30 지적)."""
    from airis.optimize import cli

    base = scenarios["default"]
    starts = cli.parse_starts("default,hands_up")
    kept, dropped = cli.starts_in_bounds(starts, base)
    assert [s.name for s in kept] == ["default", "hands_up"] and dropped == []

    narrowed = cli.narrow_scenario(base, ["shoulder_abduction=0,90"])
    kept, dropped = cli.starts_in_bounds(starts, narrowed)
    assert [s.name for s in kept] == ["default"] and dropped == ["hands_up"]

    only_up = cli.narrow_scenario(base, ["shoulder_abduction=170,180"])
    with pytest.raises(SystemExit):                     # 전부 범위 밖이면 멈춘다
        cli.starts_in_bounds([starts[0]], only_up)
