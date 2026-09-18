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
