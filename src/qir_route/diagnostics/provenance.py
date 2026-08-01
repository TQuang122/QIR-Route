from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from qir_route.diagnostics.firewall import assert_diagnostic_path_allowed


def sha256_file(path: Path) -> str:
    assert_diagnostic_path_allowed(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _is_ancestor(repository_root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def verify_frozen_receipts(
    repository_root: Path, expected: dict[str, str]
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative_path, expected_hash in sorted(expected.items()):
        path = (repository_root / relative_path).resolve()
        digest = sha256_file(path)
        if digest != expected_hash:
            raise RuntimeError(
                f"frozen receipt hash mismatch for {relative_path}: "
                f"expected {expected_hash}, got {digest}"
            )
        observed[relative_path] = digest
    return observed


def build_provenance_snapshot(
    repository_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    initial_commit = str(config["initial_snapshot_commit"])
    current_commit = git_head(repository_root)
    if not _is_ancestor(repository_root, initial_commit, current_commit):
        raise RuntimeError(
            "initial post-run snapshot is not an ancestor of the current HEAD"
        )
    receipt_hashes = verify_frozen_receipts(
        repository_root, dict(config["frozen_receipts"])
    )
    return {
        "schema_version": 1,
        "status": "verified",
        "diagnostic_only": True,
        "can_promote_frozen_method": False,
        "stage_a2_executed_before_initial_commit": True,
        "initial_post_run_snapshot_commit": initial_commit,
        "head_at_snapshot_time": current_commit,
        "frozen_receipt_sha256": receipt_hashes,
        "historical_receipts_modified": False,
    }


def write_provenance_snapshot(
    repository_root: Path, config: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    receipt = build_provenance_snapshot(repository_root, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt
