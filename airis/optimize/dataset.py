"""회귀용 데이터셋 생성. 소유자: C.

스키마 (parquet 한 행 = 체형 1개 × 시나리오 1개):
  body_*         : BodyParams 필드
  scenario       : str
  pose_*         : 최적 PoseParams 필드
  nozzle_*       : 최적 노즐 벡터 (평탄화)
  score, total_removal, discomfort
  exp_id, commit, seed

TODO(C, 3~4주차): 체형 샘플링 분포 정의, multiprocessing/GPU 배치 실행, 재개 가능하게.
"""
