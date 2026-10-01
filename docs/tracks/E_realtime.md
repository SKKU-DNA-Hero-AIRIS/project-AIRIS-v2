# 트랙 E: 실시간·시각화·발표

**목표**: (1) 최적 자세를 사람에게 보여주는 3D 마네킹 안내 뷰, (2) 입자 시뮬레이션 애니메이션, (3) 카메라 한 대로 체형을 추정하는 포즈 추정 파이프라인, (4) 이 셋을 묶은 Streamlit 대시보드(데모), (5) 발표 자료·그림. README 1절의 "게이트 앞: 체형 추정과 시나리오 선택", "실시간 응답"이 E의 몫이다.

**파일**: `airis/realtime/*`, `airis/viz/*` (단, `airis/viz/debug3d.py`는 B 소유), `scripts/run_dashboard.py`, `scripts/render_frames.py`, `tests/test_realtime.py`, `tests/test_viz.py`, `docs/figures/*`

**의존성**: B의 `build_body`·`load_nozzles`(있음), A의 프레임 덤프(단계 12, 곧 병합), C의 데이터셋·회귀 모델(4주차; 그 전까지 스텁). 물리 계산은 E가 하지 않는다 — 평가가 필요하면 D의 `PatchEvaluator`를 부른다.

**규칙**: `ti.init`을 부르지 않는다(GPU는 A 몫). `ultralytics`·`streamlit`은 함수 안에서 지연 import한다(테스트가 무겁지 않게). 카메라는 대시보드에서만 연다. 공용 파일(`types.py`, `interface.py`, `docs/interfaces.md`, `configs/`)은 수정하지 않고 PR 본문에 제안한다.

---

## 단계 1. 안내용 3D 마네킹 뷰 (`airis/viz/pose_view.py`, 1일)

사용자에게 "이 자세를 취하세요"를 보여주는 그림. B의 디버그 뷰(matplotlib 산점도)와 달리 **plotly**로 만들어 대시보드에 그대로 넣는다.

```python
def capsule_mesh(p0, p1, r, n_theta=24, n_cap=6) -> (vertices (V,3), faces (F,3))
def figure_from_state(state: BodyState, values: np.ndarray | None = None,
                      nozzle: NozzleConfig | None = None, booth: dict | None = None,
                      title: str = "") -> plotly.graph_objects.Figure
def figure_from_pose(body: BodyParams, pose: PoseParams, scenario: Scenario, **kw) -> Figure
def figure_compare(body, poses: dict[str, PoseParams], scenario) -> Figure   # 기준 자세 vs 추천 자세 나란히
```

- 캡슐 `state.capsules (K,7)`을 메시로 그린다(`capsule_part == -1`인 휠체어 프레임은 회색 반투명). `values (N,)`가 있으면 패치를 색으로 겹친다(D의 `EvalResult.extra["removal"]`).
- 부스(`configs/nozzles.yaml booth`)를 와이어프레임으로, 슬롯 바 12개를 짧은 선분으로 그린다(`nozzle.slot_axis`, `slot_length`).
- 카메라 시점: 정면·측면·위 프리셋 버튼. 축 비율 동일.
- 검증: 세 시나리오 × 기본·만세·yaw 90이 예외 없이 그려지고, `fig.to_dict()`가 직렬화된다. PNG 저장(`kaleido`가 있으면)은 선택.

## 단계 2. 입자 애니메이션 (`airis/viz/anim.py`, `scripts/render_frames.py`, 1일)

A의 프레임 덤프(`docs/interfaces.md` "시각화용 상태 덤프": `outputs/<exp_id>/frames/<step>.npz`, 키 `pos (N,3)`, `attached (N,) bool`, `part (N,) int`, `candidate (N,) int`)를 읽어 plotly 애니메이션(프레임 슬라이더)으로 만든다.

- `load_frames(dir) -> list[(step, dict)]`, `animation(frames, state, booth, candidate=0, max_points=5000) -> Figure`. 입자가 많으면 균등 서브샘플.
- 부착(회색)·부유(부위 색)·제거(빨강, 마지막 위치)로 구분. 몸은 단계 1의 메시.
- A 병합 전에는 **합성 프레임**(패치 위에서 시작해 +y로 날아가는 가짜 데이터)으로 개발·테스트한다. 로더는 스키마만 본다.
- `scripts/render_frames.py --exp <exp_id> --out outputs/<exp_id>/anim.html`.

## 단계 3. 포즈 추정 → 체형 (`airis/realtime/pose_estimate.py`, `airis/realtime/camera.py`, 2일)

- **v1 재사용**: `C:\Users\SEONGWOO\Documents\AIRIS`(project-AIRIS-MVP)의 웹캠 입력·YOLO 사람 검출·Streamlit 틀을 가져온다. 생체·시설 맥락 모듈은 쓰지 않는다.
- YOLO11-pose(`ultralytics`)로 COCO 17 키포인트를 뽑는다. 모델 가중치는 `data/models/`(gitignore)에 두고 첫 실행 때 내려받는다.
- `keypoints_to_body(kp (17,2) px, conf (17,), scale_m_per_px, ...) -> BodyParams`:
  - 키 = 발목–머리 상단(코 + 머리 반지름 보정), 어깨 너비 = 좌우 어깨 거리, 팔 길이 = 어깨–팔꿈치 + 팔꿈치–손목, 다리 길이 = 엉덩이–발목. 몸통 두께는 정면 카메라로 못 재므로 `0.13 × 키`(C 데이터셋 샘플러와 같은 비율)로 둔다.
  - 좌우 평균, 신뢰도 낮은 점은 반대쪽으로 대체. 화면 밖 키포인트면 `None`을 돌려주고 안내 문구.
  - **거리 보정**: (a) 사용자가 키를 입력하면 그 키로 px→m 스케일을 잡는다(1차), (b) 바닥 기준 마커(알려진 크기)가 보이면 그것으로(2차, 선택).
  - 여러 프레임(예: 1초분) 중앙값으로 안정화.
- 시나리오는 영상으로 판별하지 않는다(README: 정확도·윤리). 화면에서 사용자가 고른다.
- 검증(`tests/test_realtime.py`): B의 `joint_positions`로 만든 **합성 키포인트**(기본 체형을 정면 카메라에 투영)를 넣으면 키·어깨·팔·다리를 ±5% 안에 되찾는다. 실제 카메라는 테스트에서 열지 않는다.

## 단계 4. 추천 자세 연결과 대시보드 (`airis/realtime/recommend.py`, `scripts/run_dashboard.py`, 2일)

- `recommend_pose(body: BodyParams, scenario: Scenario) -> PoseParams`: C의 회귀 모델(`airis/model/`, 4주차)이 오기 전까지는 **스텁**: `docs/experiments.md`의 시나리오별 최적 자세(예: default = yaw 90·팔 내림)를 표로 두고 돌려준다. 회귀 모델 인터페이스는 C와 합의해 `docs/interfaces.md`에 제안한다(제안 형식: `predict_pose(body, scenario) -> PoseParams`, 학습 산출물 경로 `data/models/pose_regressor.joblib`).
- 대시보드(Streamlit + plotly) 화면 순서: ① 카메라/영상 파일 선택 → ② 키포인트 오버레이와 추정 체형 5개(수정 가능) → ③ 시나리오 선택(default/pregnant/wheelchair) → ④ 추천 자세 3D 마네킹 + 기준 자세(B0·B1·B2)와의 점수 비교(D `PatchEvaluator`, 400/m², 약 30 ms × 4) → ⑤ (있으면) 입자 애니메이션.
- 웹캠이 없는 환경을 위해 영상 파일·정지 이미지 입력을 지원한다.
- 검증: `streamlit run scripts/run_dashboard.py`가 뜨고, 정지 이미지 한 장으로 ①~④가 끝까지 간다. 스크린샷을 `docs/figures/dashboard_v1.png`에 남긴다.

## 단계 5. 발표 자료·그림 (5~6주차)

- E4·E3·E2 결과 그림(C·D가 만든 CSV/PNG를 정리), 파이프라인 도식, 데모 영상(30초).
- `docs/figures/`에 두고 README 6절 결과 표를 채운다(README 수정은 통합 관리자에게 제안).

## 테스트

`tests/test_viz.py`: 캡슐 메시 정점·면 수, 세 시나리오 그림 생성, 합성 프레임 애니메이션 생성. `tests/test_realtime.py`: 합성 키포인트 → 체형 ±5%, 스케일 보정, 누락 키포인트 처리, `recommend_pose` 스텁이 세 시나리오에서 부스 안 자세를 돌려줌(D의 `outside_booth`로 확인). `ultralytics`가 없으면 그 테스트만 skip.

## 완료 기준

- [ ] `figure_from_pose`가 세 시나리오 × 대표 자세 5개를 예외 없이 그리고, 슬롯 바·부스가 함께 보인다.
- [ ] 합성 프레임으로 애니메이션이 생성되고, A의 실제 덤프(`--evaluator particle`, `dump_every > 0`)로도 돈다.
- [ ] 합성 키포인트에서 체형 5개를 ±5% 안에 복원한다.
- [ ] 대시보드가 정지 이미지 입력으로 ①~④를 끝까지 수행한다(스크린샷).
- [ ] `pytest -q` 전부 통과, skip은 `ultralytics` 미설치일 때만.

## 병합 후 알림

병합되면 C 세션에 "`recommend_pose` 스텁을 회귀 모델로 교체할 인터페이스(`predict_pose`)를 확인하라", A 세션에 "프레임 덤프를 `airis/viz/anim.py`가 읽으니 스키마를 바꾸면 알려라"고 전달한다.

## 메시 몸 전환 후 (③, 2026-09-18 총괄 확정, `docs/mesh_transition.md`)

1. `airis/viz/human_mesh.py`는 MakeHuman 렌더 비교용으로 시작했고, B가 `airis/sim/human_mesh.py`로 이관·확장한 뒤에는 sim 모듈을 import하는 얇은 시각화로 줄인다.
2. `pose_view`·`anim`: 캡슐 대신 `BodyState.mesh_vertices/mesh_faces`(또는 원본 메시)를 그린다. 부위별 제거율 색은 `mesh_face_part`로 칠한다.
3. 포즈 추정의 `BodyParams` 추정을 재정의된 5개 필드(관절 중심 기준)에 맞춘다.

## 단계 6. 운전 계획 안내와 장비 제어 출력 (`airis/realtime/plan_guide.py`, `airis/viz/plan_view.py`, `docs/plan_extension.md` 7절 5번)

계획 확장(자세 순서 + 구역 세기 + 시간)을 화면과 장비 출력까지 잇는다. 물리 계산은 하지 않고 D의 `evaluate_plan`을 부른다.

- `recommend_plan(body, scenario, model=…) -> PlanRecommendation`
  - 후보 = C의 `predict_plan_candidates`(산출물 `data/models/plan_flow.pt`, 샘플 8개, `rescore=False`) + **계획 후보표**(`PLAN_STUB_TABLE`, E7 P5 w=0.1 시드 0, 묶음 `e7k14_20261001_012825_948aed`) + **P1 제품 안내 12방향 회전**(C의 `plan_baseline`).
  - 전부 이 체형·몸 모델의 패치판(`plan_physics_cfg()`, 메시 2,000/m²)으로 재채점해 최고. P1 은 12단계라 2단계 인코더를 지나지 못하므로 모델 풀 밖에서 같은 자로 잰다. E7 에서 pregnant 는 P1 이 P5 보다 높았으므로(w=0.1: 0.4975 > 0.4532) P1 이 이기면 P1 을 그대로 안내한다.
  - 예외 규약은 `recommend` 와 같다: 산출물 없음(`FileNotFoundError`)·torch 없음(`ImportError`)·학습에 없던 시나리오(`KeyError`) → 표 + P1, 사유는 `notes`. 평가기가 후보를 못 재면(`ValueError`, 아래 알려진 문제) 그 후보만 빼고 사유를 남긴다. 쓸 후보가 없으면 P0.
- `compare_plan_with_baselines` → [추천, P0 현행 운전, P1 제품 안내]. 같은 채점기라 세로 비교가 된다. 에너지 e 는 현행 운전 = 1.
- `device_control(plan, scenario, result=…) -> dict` (`airis.device_control.v1`): 몸 기준 구역 세기, 쾌적 상한, 단계별 `speed_ratio`(노즐 12개 = `apply_zone_strengths(…).strengths`)·`flow_m3_min`(비율 × `fan.rated_flow_m3_min`)·`nozzle_zone`·`chest_wall`·안내 문장, 단계 사이 `transition` 구간, `timeline_s`. `result` 를 주면 D의 에너지·점수를 옮긴다.
  - 전환(`plan.transition_s`, 1.5 s) 동안 팬은 기본 **끔**: D가 전환 동안 제거 0·에너지 0으로 보므로 같은 가정. `transition_fans="hold"`면 앞 단계 값 유지. 단계 수가 `plan.n_phases`보다 많은 계획(P1)은 이어서 도는 회전이라 전환이 없다.
- 그림: `figure_plan_phases`(단계별 3D, 4칸 초과면 고르게 4개), `figure_zone_strengths`(구역 세기 + 쾌적 상한 + 장비 상한), `figure_timeline`(단계·전환 시각표).
- 대시보드 ⑤: 단계별 안내, P0·P1 대비, 에너지·분사 시간, 3D, 구역 세기, 시각표, 비교 표, 장비 JSON 내려받기. 토글로 끌 수 있다(기존 ⑥ 애니메이션은 번호만 바뀜).
- `recommend_plan` 응답 시간(메시 2,000/m², 계획 모델 없음 = 후보 2개, 이 브랜치 측정, Apple M2 CPU): default 0.9 s, pregnant 0.3 s, wheelchair 0.6 s. P0·P1 비교는 별도로 든다.

**알려진 문제 (B에 전달)**: 캡슐 몸 휠체어에서 체형 `BodyParams(1.70, 0.42, 0.22, 0.62, 0.85)` + yaw −30° 이면 패치 수가 1,062 → 1,071 로 바뀌어 `evaluate_plan` 이 `ValueError`("단계마다 패치 수가 다르다")를 낸다. 같은 체형의 다른 yaw, 메시 몸, 다른 시나리오에서는 재현되지 않았다. E는 그 후보만 빼고 화면을 유지한다(`tests/test_plan_guide.py::test_unevaluable_candidate_is_dropped_with_note`).

테스트: `tests/test_plan_guide.py` (표가 계획 범위 안·부스 안, 최고점 선택, P1 우선, 모델 후보 사용, 폴백, 장비 JSON 의 속도 = 구역 매핑·가슴 쪽 벽 전환·시각표·풍량 한도, 그림 직렬화), `tests/test_realtime.py` 대시보드 테스트에 ⑤ 지표·JSON·토글.
