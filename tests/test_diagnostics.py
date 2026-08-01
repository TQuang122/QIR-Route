from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from qir_route.diagnostics.analysis import (
    build_slice_analyses,
    choose_verdict,
    summarize_slice,
)
from qir_route.diagnostics.firewall import (
    FirewallViolation,
    verify_test_firewall,
)
from qir_route.diagnostics.provenance import write_provenance_snapshot


def _diagnostic_frame(query_count: int, seed_count: int, delta: float) -> pd.DataFrame:
    rows = []
    for query_index in range(query_count):
        for seed_index in range(seed_count):
            rows.append(
                {
                    "query_id": f"q-{query_index}",
                    "seed": seed_index,
                    "delta_qi_vs_base": delta,
                    "delta_qi_vs_classical": delta / 2,
                    "natural_positive_in_top50": True,
                    "base_entropy_quartile": "q1",
                    "base_margin_quartile": "q1",
                    "qi_correction_magnitude_quartile": "q1",
                    "first_relevant_rank_bucket": "rank_1",
                    "source_document_group": f"g-{query_index}",
                }
            )
    return pd.DataFrame(rows)


def test_firewall_rejects_test_artifact_without_reading_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = {
        "test_firewall_intact": True,
        "split_receipt": {
            "test_access_status": "assignment_only",
            "test_candidate_cache_created": False,
            "test_metrics_computed": False,
        },
    }
    (tmp_path / "stage_a2_receipt.json").write_text(json.dumps(receipt))
    sentinel = tmp_path / "test_candidates.npz"
    sentinel.write_bytes(b"must-not-open")
    opened: list[Path] = []
    original = Path.read_bytes

    def tracked_read_bytes(path: Path) -> bytes:
        opened.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)
    with pytest.raises(FirewallViolation, match="forbidden test artifacts"):
        verify_test_firewall(tmp_path)
    assert sentinel not in opened


def test_slice_calculation_is_deterministic_and_paired_by_query_seed() -> None:
    frame = _diagnostic_frame(120, 5, 0.1)
    config = {"replicates": 500, "confidence": 0.95, "seed": 7}
    first = summarize_slice(
        frame,
        slice_family="all",
        slice_value="all",
        minimum_support=100,
        required_consistent_seeds=4,
        bootstrap_config=config,
    )
    second = summarize_slice(
        frame.sample(frac=1, random_state=9),
        slice_family="all",
        slice_value="all",
        minimum_support=100,
        required_consistent_seeds=4,
        bootstrap_config=config,
    )
    assert first == second
    assert first["query_count"] == 120
    assert first["query_seed_row_count"] == 600
    assert first["positive_direction_seed_count"] == 5
    assert first["stable_qi_regime"] is True


def test_small_support_slice_cannot_be_stable() -> None:
    frame = _diagnostic_frame(99, 5, 0.5)
    result = summarize_slice(
        frame,
        slice_family="small",
        slice_value="small",
        minimum_support=100,
        required_consistent_seeds=4,
        bootstrap_config={"replicates": 300, "confidence": 0.95, "seed": 3},
    )
    assert result["paired_bootstrap_ci"]["lower"] > 0
    assert result["support_warning"] is True
    assert result["stable_qi_regime"] is False


def test_unavailable_required_features_force_insufficient_evidence() -> None:
    frame = _diagnostic_frame(120, 5, -0.1)
    slices = build_slice_analyses(
        frame,
        minimum_support=100,
        required_consistent_seeds=4,
        bootstrap_config={"replicates": 300, "confidence": 0.95, "seed": 3},
    )
    verdict, strongest = choose_verdict(slices, ["query_length_bucket"])
    assert verdict == "insufficient_evidence"
    assert strongest is None


def test_provenance_receipt_points_to_commit_without_modifying_receipt(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test"], check=True
    )
    frozen = repository / "frozen_receipt.json"
    frozen.write_text('{"status":"verified"}\n')
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "snapshot"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    before = frozen.read_bytes()
    config = {
        "initial_snapshot_commit": commit,
        "frozen_receipts": {"frozen_receipt.json": hashlib.sha256(before).hexdigest()},
    }
    output = tmp_path / "provenance.json"
    provenance = write_provenance_snapshot(repository, config, output)
    assert provenance["initial_post_run_snapshot_commit"] == commit
    assert provenance["historical_receipts_modified"] is False
    assert frozen.read_bytes() == before
