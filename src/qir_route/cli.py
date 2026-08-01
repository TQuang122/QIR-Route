from __future__ import annotations

import argparse
import json
from pathlib import Path

from qir_route.baseline import run_smoke
from qir_route.quantum.benchmark import benchmark_head
from qir_route.stage_a import (
    run_stage_a1_ablation,
    run_stage_a2_confirmation,
    run_stage_a_smoke,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qir")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser(
        "smoke-baseline",
        help="run the receipt-backed CSConDa Stage 0 smoke baseline",
    )
    smoke.add_argument("--config", type=Path, required=True)
    benchmark = subparsers.add_parser(
        "benchmark-head",
        help="benchmark the 6,912-parameter QI head on aligned top-k candidates",
    )
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--candidates", type=int, default=50)
    benchmark.add_argument("--batch-size", type=int, default=1)
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--iterations", type=int, default=10)
    benchmark.add_argument("--output", type=Path)
    stage_a = subparsers.add_parser(
        "stage-a-smoke",
        help="prepare leak-controlled candidates and run listwise QI smoke training",
    )
    stage_a.add_argument("--config", type=Path, required=True)
    stage_a1 = subparsers.add_parser(
        "stage-a1-ablation",
        help="run three-seed QI residual and matched-classical ablations",
    )
    stage_a1.add_argument("--config", type=Path, required=True)
    stage_a2 = subparsers.add_parser(
        "stage-a2-confirm",
        help="run full-corpus five-seed confirmation with paired bootstrap gates",
    )
    stage_a2.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "smoke-baseline":
        run_dir = run_smoke(args.config)
        print(run_dir)
    elif args.command == "benchmark-head":
        report = benchmark_head(
            device=args.device,
            candidates=args.candidates,
            batch_size=args.batch_size,
            warmup=args.warmup,
            iterations=args.iterations,
            output=args.output,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "stage-a-smoke":
        run_dir = run_stage_a_smoke(args.config)
        print(run_dir)
    elif args.command == "stage-a1-ablation":
        run_dir = run_stage_a1_ablation(args.config)
        print(run_dir)
    elif args.command == "stage-a2-confirm":
        run_dir = run_stage_a2_confirmation(args.config)
        print(run_dir)


if __name__ == "__main__":
    main()
