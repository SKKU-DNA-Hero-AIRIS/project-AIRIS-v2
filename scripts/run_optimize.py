"""사용 예 (C가 채움):
python scripts/run_optimize.py --scenario default --evaluator patch --max-evals 2000
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="default")
    ap.add_argument("--evaluator", choices=["patch", "particle"], default="patch")
    ap.add_argument("--max-evals", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    raise NotImplementedError("C: 1주차 구현 대상")


if __name__ == "__main__":
    main()
