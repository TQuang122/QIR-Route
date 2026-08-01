from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FirewallViolation(RuntimeError):
    pass


def assert_diagnostic_path_allowed(path: Path) -> None:
    lowered = path.name.casefold()
    forbidden_names = {
        "split_assignments.jsonl",
        "test_candidates.npz",
        "test_candidates.manifest.json",
        "test_cache.npz",
        "test_metrics.json",
    }
    if lowered in forbidden_names:
        raise FirewallViolation(f"diagnostics may not access {path.name}")
    if "test" in lowered and any(
        marker in lowered for marker in ("candidate", "cache", "metric", "hash")
    ):
        raise FirewallViolation(f"diagnostics may not access {path.name}")


def read_allowed_json(path: Path) -> dict[str, Any]:
    assert_diagnostic_path_allowed(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object at {path}")
    return value


def verify_test_firewall(stage_a2_run: Path) -> dict[str, Any]:
    stage_a2_run = stage_a2_run.resolve()
    receipt_path = stage_a2_run / "stage_a2_receipt.json"
    receipt = read_allowed_json(receipt_path)
    split = receipt.get("split_receipt", {})
    violations: list[str] = []
    if receipt.get("test_firewall_intact") is not True:
        violations.append("stage_a2 receipt does not assert an intact firewall")
    if split.get("test_access_status") != "assignment_only":
        violations.append("test access status is not assignment_only")
    if split.get("test_candidate_cache_created") is not False:
        violations.append("test candidate cache was created")
    if split.get("test_metrics_computed") is not False:
        violations.append("test metrics were computed")
    forbidden_files = []
    for child in stage_a2_run.iterdir():
        lowered = child.name.casefold()
        if "test" in lowered and any(
            marker in lowered for marker in ("candidate", "cache", "metric")
        ):
            forbidden_files.append(child.name)
    if forbidden_files:
        violations.append(f"forbidden test artifacts exist: {sorted(forbidden_files)}")
    if violations:
        raise FirewallViolation("; ".join(violations))
    return {
        "status": "verified",
        "diagnostic_only": True,
        "stage_a2_receipt": str(receipt_path),
        "test_access_status": "assignment_only",
        "test_candidate_cache_created": False,
        "test_metrics_computed": False,
        "forbidden_test_artifacts": [],
        "test_data_opened": False,
    }
