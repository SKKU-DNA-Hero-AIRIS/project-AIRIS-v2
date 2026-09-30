"""데이터셋 → kNN 후보 표(`data/models/pose_knn.parquet`)를 만든다. 소유자: F.

혼합 추천(총괄 2026-09-30)의 kNN 쪽 산출물이다. `airis/model/predict.py` 가 읽는다.

사용 예:
    python scripts/build_pose_knn.py --dataset data/datasets/pose_dataset_mesh_cand.parquet
    python scripts/build_pose_knn.py --dataset ... --exclude-bodies 3 7 11   # 5-fold 평가용

표에는 체형·시나리오·최적 자세와 설정 도장(노즐·물리 해시, 몸 모델, 패치 밀도, 커밋)이 들어간다.
설정이 바뀌면 predict.py 가 로드할 때 경고한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airis.model.knn import STAMP_COLS, PoseKNN           # noqa: E402
from airis.optimize import cli                            # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="kNN 후보 표 만들기")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=str(ROOT / "data" / "models" / "pose_knn.parquet"))
    ap.add_argument("--exclude-bodies", nargs="*", type=int, default=[],
                    help="표에서 뺄 body_idx (holdout 평가용)")
    return ap


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)

    import pandas as pd

    df = pd.read_parquet(args.dataset)
    if args.exclude_bodies:
        df = df[~df["body_idx"].isin(args.exclude_bodies)]
    if df.empty:
        print("표에 넣을 행이 없다", file=sys.stderr)
        return 2
    for col in STAMP_COLS:                       # 설정이 섞인 데이터셋은 거부한다
        if col in df.columns and df[col].nunique(dropna=False) > 1:
            print(f"데이터셋의 {col} 값이 여러 개다: {sorted(map(str, df[col].unique()))}", file=sys.stderr)
            return 2

    table = PoseKNN.from_dataset(df)
    out = table.save(args.out)
    n_bodies = df["body_idx"].nunique()
    print(f"행 {len(table.table)} (체형 {n_bodies}, 시나리오 {', '.join(table.scenario_names)})")
    print(f"도장 {', '.join(f'{k}={v}' for k, v in table.meta.items())}")
    print(f"저장 위치 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
