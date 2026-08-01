from __future__ import annotations

import argparse
import json
from pathlib import Path

from qir_route.baseline import run_smoke
from qir_route.diagnostics import (
    run_post_a2_diagnostics,
    verify_test_firewall,
    write_provenance_snapshot,
)
from qir_route.quantum.benchmark import benchmark_head
from qir_route.retrieval_diagnostics import run_candidate_ceiling_audit
from qir_route.stage_a import (
    run_stage_a1_ablation,
    run_stage_a2_confirmation,
    run_stage_a_smoke,
)
from qir_route.stage_c0 import (
    dry_run_stage_c0,
    run_stage_c0,
    run_synthetic_stage_c0_smoke,
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
    diagnostics = subparsers.add_parser(
        "diagnose-stage-a2",
        help="run diagnostic-only analysis over frozen train/validation exports",
    )
    diagnostics.add_argument("--config", type=Path, required=True)
    candidate_ceiling = subparsers.add_parser(
        "audit-candidate-ceiling",
        help="audit fixed candidate-ceiling recovery strategies on frozen validation embeddings",
    )
    candidate_ceiling.add_argument("--config", type=Path, required=True)
    firewall = subparsers.add_parser(
        "verify-test-firewall",
        help="verify that the Stage A.2 test split remains assignment-only",
    )
    firewall.add_argument("--stage-a2-run", type=Path, required=True)
    provenance = subparsers.add_parser(
        "provenance-snapshot",
        help="write a post-run source and frozen-receipt provenance receipt",
    )
    provenance.add_argument("--config", type=Path, required=True)
    provenance.add_argument("--output", type=Path, required=True)
    stage_c0 = subparsers.add_parser(
        "stage-c0",
        help="run the preregistered EViRAL data and candidate-ceiling gate",
    )
    stage_c0.add_argument("--config", type=Path, required=True)
    stage_c0.add_argument("--audit-labels", type=Path)
    stage_c0.add_argument("--dry-run", action="store_true")
    stage_c0.add_argument("--synthetic-smoke", action="store_true")
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
    elif args.command == "diagnose-stage-a2":
        output_dir = run_post_a2_diagnostics(args.config)
        print(output_dir)
        receipt = json.loads(
            (output_dir / "diagnostic_receipt.json").read_text(encoding="utf-8")
        )
        print(
            json.dumps(
                {
                    "diagnostic_verdict": receipt["verdict"],
                    "stable_qi_helpful_regime_exists": receipt[
                        "stable_qi_regime_exists"
                    ],
                    "strongest_valid_slice": receipt["strongest_valid_slice"],
                    "stage_a3_scientifically_justified": receipt[
                        "stage_a3_scientifically_justified"
                    ],
                    "test_remained_untouched": receipt["test_remained_untouched"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "audit-candidate-ceiling":
        output_dir = run_candidate_ceiling_audit(args.config)
        print(output_dir)
        receipt = json.loads(
            (output_dir / "candidate_ceiling_receipt.json").read_text(encoding="utf-8")
        )
        print(
            json.dumps(
                {
                    "candidate_ceiling_verdict": receipt["verdict"],
                    "best_fixed_strategy": receipt["best_fixed_strategy"],
                    "recall_improvement": receipt["absolute_recall_improvement"],
                    "missing_queries_recovered_percentage": receipt[
                        "recovered_missing_query_percentage"
                    ],
                    "stage_b1_justified": receipt["stage_b1_justified"],
                    "stage_a3_justified": receipt["stage_a3_justified"],
                    "test_remained_untouched": receipt["test_remained_untouched"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "verify-test-firewall":
        print(
            json.dumps(
                verify_test_firewall(args.stage_a2_run),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "provenance-snapshot":
        config_path = args.config.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        receipt = write_provenance_snapshot(
            config_path.parent.parent, config, args.output.resolve()
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
    elif args.command == "stage-c0":
        if args.dry_run and args.synthetic_smoke:
            raise SystemExit("choose exactly one of --dry-run or --synthetic-smoke")
        if args.dry_run:
            print(
                json.dumps(dry_run_stage_c0(args.config), ensure_ascii=False, indent=2)
            )
        elif args.synthetic_smoke:
            print(
                json.dumps(
                    run_synthetic_stage_c0_smoke(args.config),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            if args.audit_labels is None:
                raise SystemExit("full Stage C.0 requires --audit-labels JSONL")
            print(run_stage_c0(args.config, args.audit_labels))


if __name__ == "__main__":
    main()
