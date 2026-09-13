from airis.sim.scenario import load_scenarios, load_physics


def test_scenarios_load_and_inherit():
    s = load_scenarios()
    assert set(s) >= {"default", "pregnant", "wheelchair"}
    # 상속: pregnant는 default의 bounds를 물려받고 torso_pitch만 덮어씀
    assert s["pregnant"].pose_bounds["shoulder_abduction"] == s["default"].pose_bounds["shoulder_abduction"]
    assert s["pregnant"].pose_bounds["torso_pitch"] == (-10, 15)
    assert s["wheelchair"].fixed_pose["hip_flexion"] == 90
    assert s["wheelchair"].seat_height_m == 0.5


def test_physics_config_has_required_sections():
    p = load_physics()
    for key in ("jet", "particles", "adhesion", "simulation", "scoring"):
        assert key in p
