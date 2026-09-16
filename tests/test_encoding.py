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
