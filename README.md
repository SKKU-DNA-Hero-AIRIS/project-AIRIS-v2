# project-AIRIS-v2

에어게이트를 통과하는 사람의 **체형과 시나리오(일반·임산부·휠체어)** 에 맞춰, 먼지가 가장 잘 제거되는 **자세와 노즐 높이·방향**을 찾는 프로젝트.

## 접근

```
체형·시나리오 샘플  →  입자 시뮬레이터 (Taichi GPU)  →  CMA-ES 최적화  →  (체형, 시나리오, 최적 설정) 데이터셋
                                                                                    ↓
카메라 → 포즈 추정 → 체형·시나리오  →  회귀 모델 (scikit-learn)  →  자세 안내 + 노즐 명령
```

- **시뮬레이터**: 해석적 제트장 + 충돌 제트 보정 위에서 라그랑주 입자를 굴린다. 입자 크기와 부착력 분포로 옷감을 근사한다. 후보 수백 개를 GPU에서 동시에 평가한다.
- **최적화**: CMA-ES. 시나리오별 자세 제약(관절 각도 범위, 고정 자세, 불편도 페널티)은 `configs/scenarios.yaml`.
- **검증**: 패치 기반 기준선과 비교, 물리 상수 민감도 실험, 상위 후보를 LBM 유체 솔버로 재평가.
- **강화학습을 쓰지 않는 이유**: 정적 자세 하나를 찾는 문제는 순차 의사결정이 아니므로 블랙박스 최적화가 더 단순하고 확실하다. 머신러닝은 최적화 결과를 새로운 체형에 일반화하는 회귀 단계에서 쓴다.

## 구조

```
airis/
  sim/
    types.py           # 공용 데이터 타입 (변경 시 전원 리뷰)
    interface.py       # Evaluator 추상 클래스
    body.py            # 마네킹 기하                     [B]
    jet.py             # 제트 속도장                     [B]
    scenario.py        # 시나리오·물리 상수 로더          [B]
    patch_baseline.py  # 패치 기반 빠른 평가기 (기준선)   [D]
    particles.py       # Taichi 입자 평가기               [A]
  optimize/
    cmaes_runner.py    # CMA-ES 루프                      [C]
    dataset.py         # 회귀용 데이터셋 생성              [C]
  model/               # 회귀 모델                        [C]
  viz/                 # 3D 시각화, 대시보드               [E]
  realtime/            # 포즈 추정 → 체형·시나리오          [E]
configs/
  physics.yaml         # 물리 상수 단일 소스              [B]
  scenarios.yaml       # 시나리오 제약                    [B]
validation/lbm/        # LBM 검증 파이프라인               [D]
tests/                 # 정성 검증 테스트 = CI 게이트       [D]
scripts/               # 실행 스크립트                    [C]
docs/
  interfaces.md        # 인터페이스 계약
  plan.md              # 6주 계획, 역할, 체크포인트
```

## 시작하기

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows
pip install -r requirements.txt
pip install -e .
pytest -q
```

Taichi GPU 백엔드 확인:

```bash
python -c "import taichi as ti; ti.init(arch=ti.gpu); print(ti.lang.impl.current_cfg().arch)"
```

## 문서

- [docs/interfaces.md](docs/interfaces.md): 타입·함수 계약, 데이터셋 스키마
- [docs/plan.md](docs/plan.md): 역할 분담, 주차별 계획, 체크포인트, 협업 규칙

## 참고 문헌

- Keedy et al. 2008, *Measurements of Air Jet Removal Efficiencies of Spherical Particles from Cloth and Planar Surfaces*, Aerosol Sci. Tech.
- Young, Hargather, Settles 2013, *Shear stress and particle removal measurements of a round turbulent air jet impinging normally upon a planar wall*, J. Aerosol Sci.
- Reeks & Hall 2001, rock'n'roll resuspension model, J. Aerosol Sci.
- Hansen 2016, *The CMA Evolution Strategy: A Tutorial*, arXiv:1604.00772
- Taichi-LBM3D, Fluids 2022
