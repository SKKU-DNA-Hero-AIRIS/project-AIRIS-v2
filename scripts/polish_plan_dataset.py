"""계획 데이터셋 다듬기 한 바퀴 (이웃 계획에서 다시 출발). 소유자: C.

F 의 5-fold(10-11)에서 **이웃 체형의 계획이 그 체형의 최적보다 높게 채점되는 행이 90.7%**
(2% 넘게 높은 행 53.3%, 비율 중앙 1.022)였다. 행별 최적이 예산 소진(272/300)으로 2~3% 덜
수렴했다는 뜻이다. 이 스크립트는 행마다 이웃 계획을 재채점해 가장 좋은 것과 원래 계획을
따뜻한 시작으로 CMA-ES 를 한 바퀴 더 돌고, 결과를 **새 열로** 더해 새 파일에 쓴다.

    python scripts/polish_plan_dataset.py --in data/datasets/plan_dataset_n9.parquet \
        --out data/datasets/plan_dataset_n9_polished.parquet --processes 12
    python scripts/polish_plan_dataset.py --in <파일> --out <파일> --scenarios wheelchair

원래 열(`score` · `plan_*` · `cand_*`)은 바꾸지 않는다. 더하는 열은 `polish_score`,
`polish_plan_raw`(대칭 정규화 뒤 원래 단위 8N+5), `polish_source`, `polish_gain`,
`polish_own_score`, `polish_n_evals`, `polish_starts`, `polish_max_evals`,
`polish_k_neighbours`, `polish_elapsed_s` 다. `polish_score >= score` 는 구조적으로 보장된다
(원래 계획도 후보에 넣고 최대를 고른다).

**측정과 목표를 섞지 말 것**: 다듬은 열은 다른 행(다른 폴드일 수 있다)의 계획에서 출발해 얻은
값이다. "이웃에서 옮겨 쓸 수 있는가"를 재는 측정에는 원래 `score` 를 쓰고, 학습 목표에만
`polish_*` 를 쓴다.

비용(기준 노트북 실측): 평가 1회 약 0.5 s(12프로세스 동시) → 예산 3,000 이면 행당 약 25분,
300행이면 한 대로 약 10시간이다. **BLAS 스레드를 1 로 묶고** 띄운다
(`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`). 중단되면 같은 명령을 다시
돌린다 — 이미 다듬어진 행은 건너뛴다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airis.optimize import cli                                      # noqa: E402
from airis.optimize.plan_dataset import (                           # noqa: E402
    PolishConfig, polish_plan_dataset,
)
from airis.sim.scenario import load_scenarios                       # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="계획 데이터셋 다듬기 한 바퀴")
    ap.add_argument("--in", dest="in_path", required=True, help="다듬을 계획 데이터셋")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="결과 파일 (입력과 달라야 한다 — F 가 입력을 읽는 중일 수 있다)")
    ap.add_argument("--scenarios", nargs="*", default=None, help="이 시나리오만 다듬는다")
    ap.add_argument("--k-neighbours", type=int, default=16, help="재채점할 이웃 수 (평가 1회씩)")
    ap.add_argument("--n-starts", type=int, default=1,
                    help="이웃 중 따뜻한 시작으로 쓸 개수 (원래 계획은 항상 더한다)")
    ap.add_argument("--max-evals", type=int, default=3000,
                    help="시작점들이 **나눠 쓰는** 총 예산 (시작 2개면 각 1,500)")
    ap.add_argument("--sigma0", type=float, default=0.2, help="따뜻한 시작의 초기 스텝")
    ap.add_argument("--seed", type=int, default=7000, help="CMA-ES 시드 기준값")
    ap.add_argument("--processes", type=int, default=12,
                    help="BLAS 스레드를 1 로 묶고 띄우면 12 가 가장 빠르다")
    ap.add_argument("--flush-every", type=int, default=10, help="몇 행마다 저장할지")
    return ap


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)

    if args.scenarios:
        all_scenarios = load_scenarios()
        missing = [n for n in args.scenarios if n not in all_scenarios]
        if missing:
            print(f"없는 시나리오: {missing} (가능: {', '.join(all_scenarios)})", file=sys.stderr)
            return 2

    pol = PolishConfig(k_neighbours=args.k_neighbours, n_starts=args.n_starts,
                       max_evals=args.max_evals, sigma0=args.sigma0, seed=args.seed)
    try:
        out = polish_plan_dataset(args.in_path, args.out_path, pol=pol,
                                  scenarios=args.scenarios, processes=args.processes,
                                  flush_every=args.flush_every)
    except (ValueError, cli.TrackNotMerged) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"\n행 {out['n_rows']} (새로 다듬음 {out['n_new']}, 누적 {out['n_done']})")
    print(f"올라간 행 {out['gain_rows']}개, 평균 상승 {out['gain_mean']:+.4f}")
    print(f"저장 위치    {out['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
