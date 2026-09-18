# 트랙 B: 마네킹·노즐·제트

**목표**: 체형과 자세를 받아 패치와 캡슐로 표현된 마네킹을 만들고, 고정 노즐 배치에서 임의의 점의 공기 속도를 계산한다. 다른 세 트랙의 실제 입력이 여기서 나오므로 가장 먼저 시작하고 가장 먼저 병합한다.

**파일**: `airis/sim/body.py`, `airis/sim/jet.py`, `airis/sim/scenario.py`, `airis/viz/debug3d.py`, `configs/*.yaml`, `tests/test_body.py`, `tests/test_jet.py`

**의존성**: 없음. `types.py`와 `configs/`만 있으면 시작 가능.

**공유 수식**: `docs/tracks/00_common.md` 4.1(원형 노즐), 4.1b(슬롯 제트), 4.2b(충돌 제트 → 벽면 제트 보정)를 그대로 구현한다. 기준 장비는 퓨리움 PURIUM-10000-P(슬롯 바 12개)이며 원형 배치는 비교용으로 남긴다.

---

## 단계 1. 골격 정의와 치수 (반나절)

### 관절

17개 관절을 트리로 정의한다. 좌우 대칭이며 같은 자세 각도를 양쪽에 적용한다 (비대칭 자세는 범위 밖).

```
pelvis
├── spine → chest → neck → head_center
├── chest → shoulder_L → elbow_L → wrist_L
├── chest → shoulder_R → elbow_R → wrist_R
├── hip_L → knee_L → ankle_L
└── hip_R → knee_R → ankle_R
```

### 체형 → 치수

`BodyParams` 5개에서 다음을 유도한다. 비율은 인체 측정 평균값이며 `body.py` 상단 상수로 둔다.

| 치수 | 유도 |
|---|---|
| 머리 반지름 | `0.065 × height_m` |
| 목 길이 | `0.04 × height_m` |
| 몸통 길이 (pelvis→chest) | `height_m − leg_length_m − 2×머리반지름 − 목길이` |
| 몸통 단면 | 타원, 반축 `shoulder_width_m/2 × 0.85` (좌우), `torso_depth_m/2` (앞뒤) |
| 상완 / 전완 길이 | `0.45 × arm_length_m`, `0.55 × arm_length_m` |
| 상완 / 전완 반지름 | `0.045`, `0.038` m (고정) |
| 대퇴 / 하퇴 길이 | `0.5 × leg_length_m` 각각 |
| 대퇴 / 하퇴 반지름 | `0.075`, `0.055` m (고정) |
| 골반 폭 (hip 간격) | `0.4 × shoulder_width_m` |

## 단계 2. 순기구학 (반나절)

`PoseParams` 7개 각도로 관절 위치를 계산한다. `scipy.spatial.transform.Rotation` 사용. 회전 순서와 축을 아래로 고정한다.

| 각도 | 적용 위치 | 축 | 양의 방향 |
|---|---|---|---|
| `torso_yaw` | pelvis 기준 전신 | z | 왼쪽으로 회전 (반시계, 위에서 볼 때) |
| `torso_pitch` | pelvis 기준 상체(spine 이상) | y | 앞으로 숙임 |
| `hip_flexion` | hip 기준 대퇴 | y | 앞으로 들기 |
| `knee_flexion` | knee 기준 하퇴 | y | 뒤로 굽힘 |
| `shoulder_abduction` | shoulder 기준 상완 | x (전후 축) | 옆으로 벌림 (0 = 팔 내림, 90 = 수평, 180 = 만세) |
| `shoulder_flexion` | shoulder 기준 상완 | y | 앞으로 들기 |
| `elbow_flexion` | elbow 기준 전완 | 상완에 수직인 축 | 굽힘 |

어깨는 abduction 먼저, flexion 다음으로 합성한다. 기본 자세는 `PoseParams()`: 팔 살짝 벌림, 정면, 직립.

`scenario.fixed_pose`가 있으면 해당 각도를 덮어쓴다. `scenario.seat_height_m`이 있으면 pelvis z를 그 값으로 두고, 없으면 발바닥이 z=0에 오도록 pelvis z를 계산한다. 마네킹 x, y는 부스 중앙 `(booth.length_m/2, 0)`.

## 단계 3. 캡슐 생성 (반나절)

관절 위치에서 캡슐 `(K, 7) = [x0,y0,z0, x1,y1,z1, r]`을 만든다.

| 캡슐 | 양끝 | 반지름 | 부위 |
|---|---|---|---|
| 머리 | head_center ± 0 (구 = 길이 0 캡슐) | 머리 반지름 | head |
| 몸통 | pelvis → chest | 타원 (아래 참고) | torso_front / torso_back |
| 상완 ×2 | shoulder → elbow | 0.045 | arms |
| 전완 ×2 | elbow → wrist | 0.038 | arms |
| 대퇴 ×2 | hip → knee | 0.075 | legs |
| 하퇴 ×2 | knee → ankle | 0.055 | legs |

몸통은 캡슐 배열에는 원형 반지름(좌우 반축)으로 넣고, 패치 생성 때만 타원 단면을 쓴다. 충돌·가림 판정은 원형 근사로 충분하다.

**휠체어**: `scenario.name == "wheelchair"`이면 가림용 캡슐을 추가한다. 좌석(짧고 굵은 수평 캡슐 1개), 등받이(수직 캡슐 1개), 바퀴(y 축 방향 짧은 캡슐 2개, 반지름 0.3). 이들은 `capsule_part = -1`이고 패치를 만들지 않는다.

## 단계 4. 패치 샘플링 (반나절)

캡슐 표면을 결정론적 격자로 샘플링한다 (난수 없음).

- 원통부: 길이 방향 `n_len`, 둘레 방향 `n_theta`. 패치 면적 = `2πr·L / (n_len·n_theta)`.
- 반구부 ×2: 위도 `n_lat`, 경도 `n_theta`. 면적 가중 위도 간격 사용.
- `patches_per_m2` 인자에서 `n_*`을 결정한다. 기본 2000/m² → 성인 마네킹 해석적 표면적 약 2.2 m² → 패치 약 5,200개 (최적화 루프는 400/m², 약 1,000개).
- 법선: 원통부는 축에서 바깥 방향, 반구부는 중심에서 바깥 방향.
- 몸통 타원: 매개변수 θ에서 위치 `(a cosθ, b sinθ)`, 법선 `∝ (cosθ/a, sinθ/b)`. 부위는 법선의 몸 전방 성분이 양이면 `torso_front`, 아니면 `torso_back`.
- `patch_capsule[i]` = 그 패치가 속한 캡슐 인덱스. D의 가림 판정이 자기 캡슐을 제외할 때 쓴다.

`BodyState`에 `capsule_part (K,)`와 `patch_capsule (N,)`을 채운다. 두 필드는 `types.py`에 이미 추가되어 있다.

## 단계 5. 노즐 배치 전개 (2시간)

`scenario.load_nozzles(path=None, layout=None) -> NozzleConfig`를 구현한다. `configs/nozzles.yaml`의 `active`가 가리키는 배치를 전개한다.

- 기준 배치 `slot_bars` (퓨리움): 측면 `wall_y × z_levels` = 8개 + 상단 `x_positions × y_positions` = 4개 → 12개. 전 행 슬롯(`slot_axis`, `slot_length` 채움, 4.1b). 슬롯 축은 분사 방향에 수직.
- 원형 비교 배치 `layout` (`load_nozzles(layout="layout")`): 아래 규칙으로 16개, `slot_axis=None`.
- 위치: `(x, wall_y, z)` for x in `x_positions`, wall_y in `wall_y`, z in `z_levels` → 16개.
- 방향: 벽면 안쪽 법선 `(0, −sign(wall_y), 0)`을 z 축으로 `yaw_deg`만큼 진행 방향(+x)으로, 다시 수평축으로 `pitch_deg`만큼 아래로 회전.
- 세기: `layout.strength` 모두 동일.
- `pulse_phase`: `None`.

## 단계 6. 자유 제트 속도장 (반나절)

`jet.velocity_field(points, nozzle, t, cfg) -> (P, 3)`을 `00_common.md` 4.1 그대로 구현한다. numpy 벡터화: `points (P,3)` × 노즐 M → `(M, P)` 중간 배열.

추가로 `jet.velocity_field_per_nozzle(points, nozzle, t, cfg) -> (M, P, 3)`을 제공한다. D가 노즐별 가림 판정을 한 뒤 합산해야 하므로 필요하다. `velocity_field`는 이것의 합이다.

펄스: `cfg["jet"]["pulse"]["enabled"]`이면 `gate(t)`를 곱한다. 기본 off.

## 단계 7. 디버그 3D 뷰 (반나절)

`airis/viz/debug3d.py`:

- `plot_body(state, values=None, nozzle=None, ax=None, title="") -> ax`: 패치를 3D 산점도로 그린다. `values (N,)`가 있으면 색으로, 없으면 부위별 색. `nozzle`이 있으면 위치에 화살표(`quiver`). 축 비율 동일하게.
- `plot_jet_slice(nozzle, cfg, plane="xz", y=0.0)`: 평면 위 격자에서 속도 크기를 등고선으로 그린다. 제트가 어디에 닿는지 확인용.
- `save(fig, name)`: `outputs/debug/<name>.png` 저장.

첫날부터 매 단계 이 뷰로 확인한다. 마네킹이 누워 있거나 팔이 몸을 뚫으면 여기서 보인다.

## 단계 8. 충돌 제트 → 벽면 제트 보정 (필수, 총괄 결정 ⑩)

`00_common.md` 4.2b를 그대로 구현한다. `velocity_field_per_nozzle`·`velocity_field`가 `surface_normals (P,3)`를 받고 `cfg["jet"]["impingement"]["enabled"]`면 노즐별 `w·e_r`을 더한 값을 돌려준다. `None`이면 보정 없음. 상수는 `jet.impingement.wall_jet_gain`(k, 기본 1.0, E3 스윕 변수) 하나. 옛 키 `stagnation_radius_factor`·`wall_jet_start_factor`는 삭제됐다. 보정을 켜면 τ가 커지므로 `adhesion.fabric_roughness_factor`를 같이 잡는다(PR #27: k=1, f=0.25, 기준 자세 전신 R ≈ 7.9%).

## 단계 9. 테스트

`tests/test_body.py`:
- 기본 자세 마네킹의 최고점 z ≈ `height_m` (±3%), 최저점 z ≈ 0.
- `shoulder_abduction=90`이면 wrist z ≈ shoulder z.
- `torso_yaw=180`이면 torso_front 패치의 법선 x 성분 부호가 반전.
- 휠체어 시나리오: pelvis z ≈ `seat_height_m`, 캡슐 수가 default보다 4개 많음.
- 패치 면적 합 ≈ 캡슐 표면적 합 (±5%).

`tests/test_jet.py`:
- 노즐 축 위 점: `s ≤ L_c`에서 속도 = U0, `s = 2L_c`에서 U0/2.
- 노즐 뒤(`s < 0`)는 0.
- 반경 `σ`에서 중심의 `exp(−0.5)`배.
- `load_nozzles()` 기준 배치 12개 전 행 슬롯, `slot_axis ⊥ direction`, 측면은 안쪽·상단은 아래를 향함. `load_nozzles(layout="layout")` 원형 16개, `slot_axis=None`, 모두 부스 안쪽 (`d·(0,−sign(y),0) > 0`).
- 4.1b: 코어 끝 연속, `s = 4L_c`에서 `U0/2`, 슬롯 길이 안 균일·끝 밖 가우시안, 독립 참고 구현(float64) 대조 1e-5.
- 4.2b: 정체점 0, 고리 최대, 바깥 원형 1/ξ·슬롯 1/√ξ, `enabled=false`면 4.1과 비트 일치, 독립 참고 구현 대조 1e-4.

## 완료 기준

- [ ] `build_body(BodyParams(), PoseParams(), scenarios["default"])`가 예외 없이 `BodyState`를 반환하고 3D 뷰에서 사람 형태로 서 있다.
- [ ] 세 시나리오 모두 마네킹이 생성되고 휠체어는 앉아 있다.
- [ ] 기준 배치 슬롯 12개가 세 시나리오 마네킹에 닿는다 (몸 높이 안 측면 바는 `U_c(0.73 m)` 이상, 12개 전부 0.5 m/s 초과). 원형 비교 배치 16개도 닿는다.
- [ ] `test_body.py`, `test_jet.py` 전부 통과.
- [ ] 1회 `build_body` 20 ms 이하. `velocity_field` 무보정 패치 3,600개 × 슬롯 12개 5 ms 이하, **4.2b 보정 포함(법선 전달) 25 ms 이하**.
- [ ] `configs/physics.yaml`의 `jet` 섹션 키가 코드에서 전부 읽힌다 (안 쓰는 키 없음).
- [ ] 세 시나리오의 기본 자세 `PoseParams()`가 부스 안이다 (`00_common.md` 5절: `|y| ≤ width/2`, `z ≤ height`). 설정 변경이 기본 자세를 불가로 만들면 안 된다 (PR #21 사례).

## 병합 후 알림

병합되면 D, C, A 세션에 "B 병합됨. `tests/fakes.py`의 `fake_body`, `fake_nozzles`를 `build_body`, `load_nozzles`로 교체하라"고 전달한다.
