# 사람 메시 몸 전환 계획 (③) — 2026-09-18 확정

캡슐 마네킹을 MakeHuman(CC0) 사람 메시로 바꾼다. 총괄 결정: 메시는 MakeHuman, 설계 결정 1~10은 아래 추천안 그대로.

## 배경

- 캡슐 마네킹은 키 1.70 기준 어깨 관절 간격 0.42(메시 0.34), 팔 0.62(0.46), 고관절 간격 0.08(0.23) m로 실제 사람과 치수가 어긋나고 목·손·발이 없다. 팔이 길고 넓게 뻗고 다리가 붙어 있어 가림과 노출 면적이 사람과 다르다.
- 후보 비교(E, `outputs/mesh_preview/README.md`): MakeHuman은 자산·결과물 모두 CC0. SMPL-X는 비상업·재배포 금지라 공개 저장소·클라우드 데이터셋·퓨리움 데모 모두에 위험 → 제외.
- 자산: `data/meshes/makehuman/{base.obj, default.mhskel, default_weights.mhw}` (2.8 MB, makehumancommunity/makehuman master, CC0). 정점 13,380, 삼각형 26,756, 뼈 163(가중치 있는 뼈 139).

## 설계 결정 (확정)

1. **`BodyState` 확장** (`airis/sim/types.py`): `mesh_vertices (V,3)`, `mesh_faces (F,3)`, `mesh_face_part (F,)`, `patch_face (N,)` 추가(모두 `None` 기본). `capsules`는 유지하되 메시 모델에서는 "뼈에 맞춘 근사 캡슐"(A 입자 충돌·D 폴백·휠체어 프레임)을 담는다. `patch_capsule`은 캡슐 모델 전용.
2. **시뮬레이션용 메시 해상도**: 원본 26,756 삼각형을 오프라인 1회 데시메이션한 약 5~6k 삼각형 "sim 메시"(정점별 뼈 가중치 보존)를 계산에 쓴다. 시각화는 원본. 파생 파일도 CC0이므로 저장소에 커밋(`data/meshes/makehuman/sim_*.npz` 등, B가 생성 스크립트와 함께).
3. **자산 커밋**: CC0 원본 3개 + `LICENSE.txt`(CC0 전문·출처 URL·커밋 해시)를 저장소에 넣는다. `.gitignore`는 `data/meshes/smplx/`만 제외.
4. **모듈 소유**: 메시 로드·스키닝·체형 맞춤·패치 샘플링은 `airis/sim/human_mesh.py`(B 소유). E의 `airis/viz/human_mesh.py`(렌더 비교용)는 B 이관 후 sim 모듈을 import하는 얇은 시각화로 줄인다.
5. **가림(D)**: `open3d.t.geometry.RaycastingScene`(BSD, Windows CPU) 광선-삼각형 교차로 교체. 광선 원점은 노즐 가림점(슬롯 K=3 유지), 대상은 패치 위치, 자기 면(`patch_face`)은 제외. 캡슐 경로는 `body.model: capsule`일 때 폴백으로 유지.
6. **입자 충돌(A)**: 1차는 뼈 캡슐(`capsules`, 약 30개)로 기존 Taichi 커널 유지. 가림은 D의 `occlusion` 재사용이라 자동으로 메시. SDF 격자 전환은 E5 결과를 보고 결정.
7. **성능 기준**: 패치판 400/m² 1회 **30 ms → 100 ms**(중앙값, 저부하). 3,000회 최적화 약 5분, 데이터셋 100체형×3시나리오 약 5시간(8 프로세스).
8. **`BodyParams` 재정의**: 5개 필드를 메시 측정(관절 중심) 기준으로 재정의 — 키, 어깨 관절 간격, 가슴 두께, 어깨→손목, 고관절 높이. 기본값은 B의 메시 `build_body` PR과 같은 시점에 MakeHuman 기본 체형(1.70/0.34/0.22/0.46/0.88)으로 교체(별도 types.py PR). 체형 맞춤은 뼈 길이별 축척(1차) → 모디파이어 타깃(선택). 임산부는 MakeHuman `stomach-pregnant` 타깃(추가 다운로드, B 세션 사용자 승인) 적용. C의 데이터셋 샘플링 범위와 E의 포즈 추정 매핑을 함께 갱신.
9. **전환 검증(E5)**: 캡슐판 vs 메시판, 시나리오별 부스 안 자세 300개 Spearman(score·부위별), E2 방식. D 소유 `scripts/compare_bodies.py`. 통과 기준 없음 — 차이를 `docs/experiments.md`에 기록.
10. **결과 재실행**: 전환 후 C가 E4 → E3 → 데이터셋 순으로 재실행. 캡슐판 E3·데이터셋 결과는 "캡슐판 참고"로 보존.

## 설정

`configs/physics.yaml`에 `body.model: mesh | capsule` (B). 전환 완료 전까지 기본은 `capsule`, B·D·A PR이 모두 병합되고 E5가 기록되면 `mesh`로 바꾼다(B PR).

## 순서와 담당

| 순서 | 담당 | 내용 | 예상 |
|---|---|---|---|
| 0 | 통합 | 이 문서, `types.py` 필드, `interfaces.md`, `00_common.md` §1·§5·§7 | 즉시 |
| 1 | E | `airis/viz/human_mesh.py`(MakeHuman 전용, 희소 스키닝) + 자산 커밋 + `tests/test_viz.py` | 0.5일 |
| 2 | B | `airis/sim/human_mesh.py` 이관·확장: BodyParams→뼈 축척, 임산부 타깃, sim 메시 데시메이션, 패치 샘플링(면 중심·법선·면적·부위, 좌우 대칭), 뼈 캡슐, `build_body(model=)` 스위치, 테스트 | 4일 |
| 3 | D | 메시 레이캐스트 가림(Open3D), 성능 측정(100 ms), `scripts/compare_bodies.py`(E5) | 3일, 2와 병행(가짜 메시로 시작) |
| 4 | A | 뼈 캡슐 소비 확인, `max_capsules` 조정, 부위 판정, 테스트 | 0.5일 |
| 5 | B | E5 기록 후 `body.model: mesh` 기본값 전환, BodyParams 기본값 PR(통합) | 0.5일 |
| 6 | C | BodyParams 샘플링 범위 갱신, E4 → E3 → 데이터셋 재실행 | 재실행 1~2일 |
| 7 | E | `pose_view`·`anim` 메시 렌더, 포즈 추정 BodyParams 매핑 | 1일 |

## 패치 샘플링 규약 (B, D·A가 소비)

- 패치 = sim 메시 면 위의 점. `patches_per_m2` 밀도로 면적 비례 샘플링(면적이 큰 면은 여러 점, 작은 면은 확률적으로 0~1개). 위치는 면 위, 법선은 면 법선(바깥), 면적은 그 면의 면적을 그 면의 패치 수로 나눈 값.
- 좌우 대칭: y → −y 대칭 쌍이 되도록 샘플링(#42 규칙 유지). MakeHuman 기본 메시는 좌우 대칭이므로 면 인덱스 대칭 맵을 오프라인에서 만들어 저장.
- 부위(`PART_NAMES`): 정점 뼈 가중치 최댓값의 뼈 → 부위 매핑(머리·목 → head, 척추·골반 → torso, 상완·전완·손 → arms, 대퇴·하퇴·발 → legs). torso는 면 법선의 몸 전방 성분 부호로 front/back(기존 규칙).
- 부스 밖 판정은 패치 위치 기준(§5 규칙 그대로).

## 사용자 할 일

- MakeHuman 타깃 파일(임산부 배·체형 모디파이어) 추가 다운로드 승인(B 세션에서 요청).
- Open3D 설치 승인(D 세션에서 `pip install open3d`).
