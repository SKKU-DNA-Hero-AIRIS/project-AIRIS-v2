# 발표 자료 관계도

그림 한 장: `fig_materials_map.png` (추천 설명 순서 위, 자료 출처 아래). 아래는 같은 내용을 표로 적은 것이다. **1절의 순서는 통합의 추천안이고, 최종 발표 순서는 발표자가 정한다** — 자료가 서로 어떻게 이어지는지 보는 용도다.

## 1. 추천 설명 순서와 자료의 연결 (참고용)

| 순서 | 말하는 것 | 쓰는 자료 | 그 자료가 보여 주는 것 |
|---|---|---|---|
| ① 문제와 목표 | 체형·자세마다 바람이 닿는 곳이 다르다 → 사용자별 최적화 | `fig1_booth_nozzles.png`, `screens/pose_*_b0_*.png` | 기준 장비 부스와 분사구 12개, 안내 없이 선 자세에 바람이 닿는 모습 |
| ② 시스템 구조 | 미리 계산(최적화·학습) → 게이트 앞 실시간 추천 | `fig_system_architecture.png` | 두 단계와 그 사이에 넘어가는 것(학습된 모델) |
| ③ 바람·먼지 계산 모델 | 분사구 바람 → 몸 표면 흐름 → 문지르는 힘 → 제거 효과 | `fig2_jet_impingement_model.png`, `calibration_notes.md`(발표 노트) | 모델의 세 단계와 보정 계수의 근거·한계 |
| ④ 자세 최적화 결과 | 세 유형 모두 "팔 들고 옆으로", 제조사 안내 대비 +55~63% | `fig_e4_scores_by_scenario.png`, `fig_e4_optimal_pose_angles.png`, `screens/pose_*_{b1,rec}_*.png`, `screens/pose_default_rec_h1{60,85}.png` | 점수 비교, 자세 각도, 3D 마네킹, 체형이 바뀌면 제안이 바뀌는 예 |
| ⑤ AI 추천과 시연 | 처음 보는 체형에서도 99% 품질, 응답 약 1초, 입력 오차 ±2% 조건 | `fig_model_quality.png`, `fig_response_time.png`, `fig_input_error_robustness.png`, `screens/dashboard_1~3_*.png`, `screens/dashboard_flow.gif`, `screens/flow_01~09.png`, `screens/particles_*.gif` | 품질·속도·내성 수치, 실제 화면(뼈대 표시·추천·모델 상태), 화면 흐름, 입자 시뮬레이션 |
| ⑥ 운전 계획·향후 | 도는 횟수가 핵심, 같은 횟수면 자세 변경 계획이 +7~13%, 실시간은 "최적 자세 + 회전 안내" | `../figures/fig_e7_phase_count.png`, `fig_e7_same_count.png`, `fig_e7_rotation_pose.png`, `screens/dashboard_4_rotation.png` | 단계 수 효과, 같은 9단계 비교, 회전 기준 자세 비교, 회전 안내 화면 |
| ⑦ 한계 | 모든 장에 공통 | `README.md` "발표에서 쓰지 말 것" | 상대값·미보정·장비 연동 없음·입력 오차 조건 |

## 2. 자료의 출처 (수치는 메시지가 아니라 파일에서 읽는다)

| 그림·캡처 | 수치 원본 (저장소) | 실험 |
|---|---|---|
| `fig_e4_*` 2장, `screens/pose_*` | `docs/e4_reference.json` | 자세 최적화 실험(시나리오 3 × 무작위 시작 5) |
| `fig_e7_*` 3장 (`docs/figures/`) | `docs/e7_sweep_reference.json`, `docs/e7_reference.json` (`scripts/fig_e7_plans.py`) | 운전 계획 실험(단계 수 2·3·4·6·9, 회전 기준선, 시드 2, 입자판 검증) |
| `fig_model_quality` | `docs/experiments_model.md` 2.2 (`outputs/refresh_k14/e5cv_overall.csv`) | AI 추천 품질(체형 300개, 5-fold) |
| `fig_response_time` | `docs/experiments_model.md` 5.3 (`outputs/e5timing_20261004_summary.csv`) | 응답 시간(구성 4개, 단독 측정) |
| `fig_input_error_robustness` | `docs/experiments_model.md` 7.1·7.2 (`outputs/body_noise_k14/summary.csv`) | 체형 입력 오차 내성, 판정 여유 2% |
| `fig1`, `fig2` (B) | `configs/physics.yaml`, `configs/nozzles.yaml`, 논문값(Phares 등 2000, 그림에서 읽음) | — |
| `fig_system_architecture` | `docs/interfaces.md`, 코드 구조 | — |
| `screens/dashboard_*`, `flow_*` | 실제 대시보드(설치된 AI 모델, 설정 일치 상태) | — |
| `screens/particles_*.gif` | 정밀 입자 시뮬레이션 실제 실행 덤프 | — |
| 완성도 % | `docs/completion_criteria.md` | — |

모든 수치 파일은 설정 식별값(물리 `c5aea26d`, 분사구 `3b99200e`)이 현재 설정과 같음을 확인한 것이다.

## 3. 같은 수치가 여러 자료에 나올 때 맞춰야 하는 것

- **제조사 안내 대비 +62/+55/+63%**(`fig_e4_scores`)와 **대시보드의 +135/+57/+109%**(`dashboard_2`)는 기준이 다르다. 전자는 표준 체형·제조사 안내 대비, 후자는 예시 체형·세 기준(안내 없음/몸 돌리기/만세) 각각 대비. 캡션에 기준을 적는다.
- **AI 품질 0.993 / 0.33%**는 현재 물리 설정 값이다. 이전 설정(k=1)의 0.995 / 0.67%와 섞지 않는다.
- **응답 시간**: 측정값 평균 0.55초·95% 0.78초(`fig_response_time`)와 대시보드 캡처의 0.96초(혼잡 중 한 번)는 조건이 다르다. 발표에서는 "약 1초 안"으로 통일한다.
- **운전 계획 점수**: 단계 수 스윕(에너지 가중 0.01)과 9단계 비교(가중 0.1)는 가중이 달라 한 축에 놓지 않는다. 그래서 그림이 둘로 나뉘어 있다.
- **3D 마네킹의 "몸 돌리기" 점수**는 12방향 중 한 방향(옆)만 그린 것이라 표의 평균 점수와 다르다. 캡션 "회전 중 한 방향".
- **단계 수 상한 9**: 총 분사 시간 상한 20초 ÷ 최소 단계 2초. "왜 10이나 12가 아닌가"가 나오면 이 설정 때문이라고 답한다. 회전 안내 화면의 10방향은 자세를 바꾸지 않아 전환 시간이 없어서 가능하다.

## 4. 아직 없는 자료 (시점)

- 학습 곡선(체형 수별 AI 품질): 1,000체형 데이터셋 완료, F 계산 중(10-07) → `fig_learning_curve` 추가 예정.
- 계획 추천 모델 품질·응답(3차): 계획 데이터셋 생성 중(10-07~10-09) → 학습·판정 뒤. 발표가 그 전이면 "향후 계획"으로.
- 실제 조작 녹화 영상: 총괄 직접(대본은 저장소 밖). 실제 장비·실제 사람 검증: 범위 밖.
