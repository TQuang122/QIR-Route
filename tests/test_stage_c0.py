from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from qir_route.stage_c0.core import (
    DataAcceptanceError,
    audit_structural_data,
    choose_candidate_lane,
    evaluate_human_audit,
    evaluate_rankings,
    exact_dense_rankings,
    identifier_digest,
    partition_query_ids,
    paired_lane_intervals,
    percentile_bootstrap_interval,
    reciprocal_rank_fusion,
    select_human_audit_query_ids,
    stage_c0_verdict,
    _stable_topk,
)
from qir_route.stage_c0.pipeline import (
    StageC0FirewallViolation,
    _ensure_failure_artifacts,
    _git_implementation_lineage_frozen,
    _read_jsonl,
    _resolve_frozen_root,
    _validate_download_url,
    _validate_model_download_plan,
    assert_asset_allowed,
    dry_run_stage_c0,
    load_stage_c0_config,
    run_synthetic_stage_c0_smoke,
    run_stage_c0,
    validate_frozen_config,
    verify_stage_c0_firewall,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "stage_c0_eviral.yaml"


def synthetic_frames(size: int = 16) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    corpus = pd.DataFrame(
        {
            "corpus_id": [f"d-{index}" for index in range(size)],
            "title": [f"Title {index}" for index in range(size)],
            "passage": [f"Passage about topic {index}" for index in range(size)],
        }
    )
    queries = pd.DataFrame(
        {
            "query_id": [f"q-{index}" for index in range(size)],
            "query_vi": [f"Câu hỏi chủ đề {index}" for index in range(size)],
        }
    )
    qrels = pd.DataFrame(
        {
            "query_id": queries["query_id"],
            "corpus_id": corpus["corpus_id"],
            "score": np.ones(size, dtype=np.float64),
        }
    )
    return corpus, queries, qrels


def test_frozen_config_and_dry_run_do_not_access_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_network(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("dry-run attempted network access")

    monkeypatch.setattr("urllib.request.urlopen", fail_network)
    config = load_stage_c0_config(CONFIG_PATH)
    receipt = dry_run_stage_c0(CONFIG_PATH)
    assert config["decision_id"] == "QIR-EVIRAL-C0-001"
    expected_status = "verified" if receipt["full_run_freeze_ready"] else "blocked"
    assert receipt["status"] == expected_status
    assert receipt["network_accessed"] is False
    assert receipt["data_downloaded"] is False
    assert receipt["validation_rows_accessed"] is False
    assert receipt["test_rows_accessed"] is False
    assert receipt["preregistration_hash_matches"] is True
    assert receipt["implementation_lineage_frozen"] is True
    assert set(receipt["runner_git_frozen"]) == {
        "src/qir_route/cli.py",
        "src/qir_route/baseline.py",
        "src/qir_route/stage_c0",
        "configs/stage_c0_eviral.yaml",
        "tests/test_stage_c0.py",
    }


@pytest.mark.parametrize(
    "asset",
    [
        "queries/validation-00000-of-00001.parquet",
        "qrels/validation-00000-of-00001.parquet",
        "queries/test-00000-of-00001.parquet",
        "qrels/test-00000-of-00001.parquet",
        "queries/latest.parquet",
    ],
)
def test_asset_firewall_rejects_every_non_whitelisted_path(asset: str) -> None:
    config = load_stage_c0_config(CONFIG_PATH)
    with pytest.raises(StageC0FirewallViolation):
        assert_asset_allowed(asset, config)


def test_partition_is_exact_hash_order_and_deterministic() -> None:
    query_ids = [f"q-{index}" for index in range(20)]
    first = partition_query_ids(
        query_ids, decision_id="QIR-EVIRAL-C0-001", calibration_count=4
    )
    second = partition_query_ids(
        list(reversed(query_ids)),
        decision_id="QIR-EVIRAL-C0-001",
        calibration_count=4,
    )
    expected = sorted(
        query_ids,
        key=lambda value: (
            identifier_digest("QIR-EVIRAL-C0-001", value),
            value,
        ),
    )
    assert first == second
    assert first["calibration"] == expected[:4]
    assert first["fit"] == expected[4:]


def test_structural_audit_enforces_qrel_and_cross_partition_contracts() -> None:
    corpus, queries, qrels = synthetic_frames()
    partitions, receipt = audit_structural_data(
        corpus,
        queries,
        qrels,
        decision_id="QIR-EVIRAL-C0-001",
        calibration_count=4,
    )
    assert receipt["one_positive_qrel_per_query"] is True
    assert receipt["calibration_count"] == 4
    assert set(partitions["fit"]).isdisjoint(partitions["calibration"])

    duplicated = queries.copy()
    fit_id = partitions["fit"][0]
    calibration_id = partitions["calibration"][0]
    duplicate_text = duplicated.loc[duplicated["query_id"] == fit_id, "query_vi"].item()
    duplicated.loc[duplicated["query_id"] == calibration_id, "query_vi"] = (
        duplicate_text
    )
    with pytest.raises(DataAcceptanceError, match="duplicate queries cross"):
        audit_structural_data(
            corpus,
            duplicated,
            qrels,
            decision_id="QIR-EVIRAL-C0-001",
            calibration_count=4,
        )

    null_query = queries.copy()
    null_query.loc[0, "query_vi"] = None
    with pytest.raises(DataAcceptanceError, match="must not be null"):
        audit_structural_data(
            corpus,
            null_query,
            qrels,
            decision_id="QIR-EVIRAL-C0-001",
            calibration_count=4,
        )

    null_identifier = corpus.copy()
    null_identifier.loc[0, "corpus_id"] = None
    with pytest.raises(DataAcceptanceError, match="must not be null"):
        audit_structural_data(
            null_identifier,
            queries,
            qrels,
            decision_id="QIR-EVIRAL-C0-001",
            calibration_count=4,
        )


def test_human_audit_sample_and_labels_are_frozen_without_raw_text() -> None:
    _, queries, qrels = synthetic_frames(32)
    partitions = partition_query_ids(
        queries["query_id"].tolist(),
        decision_id="QIR-EVIRAL-C0-001",
        calibration_count=16,
    )
    selected = select_human_audit_query_ids(
        queries,
        partitions["calibration"],
        decision_id="QIR-EVIRAL-C0-001",
        samples_per_quartile=1,
    )
    assert len(selected) == 4
    gold = dict(zip(qrels["query_id"], qrels["corpus_id"], strict=True))
    records = [
        {
            "query_id": query_id,
            "corpus_id": gold[query_id],
            "label": "supported",
            "reason_codes": [],
            "review_timestamp": "2026-08-01T00:00:00Z",
            "rubric_version": "1",
            "reviewer_pseudonym": "reviewer-1",
        }
        for query_id in selected
    ]
    receipt = evaluate_human_audit(
        records,
        expected_query_ids=selected,
        gold_corpus_by_query=gold,
        minimum_supported=4,
        maximum_non_vietnamese=0,
    )
    assert receipt["passed"] is True
    assert receipt["raw_text_exported"] is False

    contaminated = [dict(record) for record in records]
    contaminated[0]["query_vi"] = "raw text is forbidden"
    with pytest.raises(DataAcceptanceError, match="raw text"):
        evaluate_human_audit(
            contaminated,
            expected_query_ids=selected,
            gold_corpus_by_query=gold,
            minimum_supported=4,
            maximum_non_vietnamese=0,
        )

    ambiguous = [dict(record) for record in records]
    ambiguous[0]["reason_codes"] = ["ambiguous"]
    with pytest.raises(DataAcceptanceError, match="ambiguous.*not_supported"):
        evaluate_human_audit(
            ambiguous,
            expected_query_ids=selected,
            gold_corpus_by_query=gold,
            minimum_supported=4,
            maximum_non_vietnamese=0,
        )

    multiple_reviewers = [dict(record) for record in records]
    multiple_reviewers[0]["reviewer_pseudonym"] = "reviewer-2"
    with pytest.raises(DataAcceptanceError, match="exactly one reviewer"):
        evaluate_human_audit(
            multiple_reviewers,
            expected_query_ids=selected,
            gold_corpus_by_query=gold,
            minimum_supported=4,
            maximum_non_vietnamese=0,
        )


def test_exact_dense_ranking_is_block_invariant_and_ties_use_document_id() -> None:
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    documents = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        dtype=np.float32,
    )
    document_ids = ["d-b", "d-a", "d-c", "d-d"]
    first, first_scores = exact_dense_rankings(
        query,
        documents,
        document_ids,
        top_k=4,
        query_block_size=1,
        document_block_size=2,
    )
    second, second_scores = exact_dense_rankings(
        query,
        documents,
        document_ids,
        top_k=4,
        query_block_size=2,
        document_block_size=4,
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first_scores, second_scores)
    assert first[0, :2].tolist() == [1, 0]


def test_topk_tie_at_partition_boundary_uses_document_id() -> None:
    indices, scores = _stable_topk(
        np.asarray([2.0, 1.0, 1.0, 1.0]),
        np.asarray(["d-z", "d-c", "d-a", "d-b"]),
        3,
    )
    assert indices.tolist() == [0, 2, 3]
    assert scores.tolist() == [2.0, 1.0, 1.0]


def test_operational_roots_and_download_urls_are_confined(tmp_path: Path) -> None:
    config = load_stage_c0_config(CONFIG_PATH)
    changed = dict(config)
    changed["cache_root"] = "../outside"
    with pytest.raises(ValueError, match="cache root differs"):
        validate_frozen_config(changed)

    (tmp_path / "artifacts").mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / "artifacts" / "stage_c0").symlink_to(external)
    with pytest.raises(StageC0FirewallViolation, match="symlink"):
        _resolve_frozen_root(tmp_path, "artifacts/stage_c0", "artifacts/stage_c0")

    _validate_download_url("https://huggingface.co/datasets/NIRVLab/EViRAL")
    _validate_download_url("https://cas-bridge.xethub.hf.co/file")
    with pytest.raises(StageC0FirewallViolation):
        _validate_download_url("http://huggingface.co/file")
    with pytest.raises(StageC0FirewallViolation):
        _validate_download_url("https://127.0.0.1/file")

    plan = [SimpleNamespace(filename="config.json", file_size=100)]
    assert _validate_model_download_plan(plan) == [
        {"filename": "config.json", "bytes": 100}
    ]
    with pytest.raises(StageC0FirewallViolation, match="path"):
        _validate_model_download_plan(
            [SimpleNamespace(filename="../escape.bin", file_size=100)]
        )

    audit_target = tmp_path / "audit-target.jsonl"
    audit_target.write_text("{}\n", encoding="utf-8")
    audit_link = tmp_path / "audit-link.jsonl"
    audit_link.symlink_to(audit_target)
    with pytest.raises(DataAcceptanceError, match="non-symlink"):
        run_stage_c0(CONFIG_PATH, audit_link)

    cache = tmp_path / "cache"
    output = tmp_path / "output"
    cache.mkdir()
    output.mkdir()
    (output / "innocent-name.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(StageC0FirewallViolation, match="forbidden.*artifacts"):
        verify_stage_c0_firewall(cache, output)

    content_output = tmp_path / "content-output"
    content_output.mkdir()
    (content_output / "candidate_metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metrics": {"bm25": {"document_text": "raw text"}},
                "selected_lane": "bm25",
                "paired_lane_intervals": {},
                "deterministic_tie_break": {
                    "document_ties": "corpus_id_ascending",
                    "lane_metric_ties": "rrf_then_dense_then_bm25",
                },
                "promotion_gate_observations": {},
                "promotion_thresholds": {
                    "minimum_recall_at_100": 0.9,
                    "minimum_bootstrap_lower": 0.88,
                },
                "secondary_metrics_have_promotion_authority": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(StageC0FirewallViolation, match="forbidden.*artifacts"):
        verify_stage_c0_firewall(cache, content_output)

    scalar_output = tmp_path / "scalar-output"
    scalar_output.mkdir()
    (scalar_output / "candidate_metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metrics": {
                    "bm25": {
                        "recall_at_10": 0.0,
                        "recall_at_20": 0.0,
                        "recall_at_50": 0.0,
                        "recall_at_100": "raw query text",
                        "mrr_at_10": 0.0,
                        "ndcg_at_10": 0.0,
                    }
                },
                "selected_lane": "bm25",
                "paired_lane_intervals": {},
                "deterministic_tie_break": {
                    "document_ties": "corpus_id_ascending",
                    "lane_metric_ties": "rrf_then_dense_then_bm25",
                },
                "promotion_gate_observations": {},
                "promotion_thresholds": {
                    "minimum_recall_at_100": 0.9,
                    "minimum_bootstrap_lower": 0.88,
                },
                "secondary_metrics_have_promotion_authority": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(StageC0FirewallViolation, match="forbidden.*artifacts"):
        verify_stage_c0_firewall(cache, scalar_output)

    raw_output = tmp_path / "raw-output"
    raw_output.mkdir()
    (raw_output / "candidate_metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "not_run",
                "reason": "secret row text",
                "selected_lane": None,
                "deterministic_tie_break": {
                    "document_ties": "corpus_id_ascending",
                    "lane_metric_ties": "rrf_then_dense_then_bm25",
                },
                "promotion_gate_observations": {"human_audit_gates_passed": False},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(StageC0FirewallViolation, match="forbidden.*artifacts"):
        verify_stage_c0_firewall(cache, raw_output)

    ranking_output = tmp_path / "ranking-output"
    ranking_output.mkdir()
    np.savez_compressed(
        ranking_output / "candidate_rankings.npz",
        query_id_sha256=np.asarray(["raw query text"]),
        bm25_document_id_sha256=np.asarray([["raw passage text"]]),
        dense_document_id_sha256=np.asarray([["raw passage text"]]),
        rrf_document_id_sha256=np.asarray([["raw passage text"]]),
    )
    with pytest.raises(StageC0FirewallViolation, match="forbidden.*artifacts"):
        verify_stage_c0_firewall(cache, ranking_output)

    arbitrary_cache = tmp_path / "arbitrary-cache"
    arbitrary_cache.mkdir()
    (arbitrary_cache / "model.bin").write_bytes(b"arbitrary")
    with pytest.raises(StageC0FirewallViolation, match="expected SHA-256"):
        verify_stage_c0_firewall(
            arbitrary_cache,
            allowed_cache_files={"model.bin"},
        )


def test_implementation_lineage_rejects_a_later_clean_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "git-repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", repository], check=True)
    subprocess.run(
        ["git", "-C", repository, "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", repository, "config", "user.name", "Test"], check=True)
    (repository / "prereg.md").write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "-C", repository, "add", "prereg.md"], check=True)
    subprocess.run(
        ["git", "-C", repository, "commit", "-q", "-m", "Preregister"],
        check=True,
    )
    parent = subprocess.run(
        ["git", "-C", repository, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(
        "qir_route.stage_c0.pipeline.FROZEN_PREREGISTRATION_COMMIT", parent
    )
    frozen_paths = (
        "configs/stage_c0_eviral.yaml",
        "src/qir_route/cli.py",
        "src/qir_route/stage_c0/__init__.py",
        "src/qir_route/stage_c0/core.py",
        "src/qir_route/stage_c0/pipeline.py",
        "tests/test_stage_c0.py",
    )
    for relative in frozen_paths:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n", encoding="utf-8")
    subprocess.run(["git", "-C", repository, "add", *frozen_paths], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            repository,
            "commit",
            "-q",
            "-m",
            "Implement preregistered EViRAL Stage C.0 runner",
        ],
        check=True,
    )
    assert _git_implementation_lineage_frozen(repository) is True
    (repository / "later.txt").write_text("later\n", encoding="utf-8")
    subprocess.run(["git", "-C", repository, "add", "later.txt"], check=True)
    subprocess.run(
        ["git", "-C", repository, "commit", "-q", "-m", "Later clean commit"],
        check=True,
    )
    assert _git_implementation_lineage_frozen(repository) is False


def test_audit_input_is_bounded_and_failure_artifacts_are_complete(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text("{}\n{}\n", encoding="utf-8")
    with pytest.raises(DataAcceptanceError, match="exceeds 200"):
        _read_jsonl(labels, expected_count=1)

    output = tmp_path / "run"
    output.mkdir()
    config = load_stage_c0_config(CONFIG_PATH)
    _ensure_failure_artifacts(
        output,
        config,
        verdict="incomplete_technical_run",
        error=RuntimeError("raw query: synthetic technical failure"),
    )
    required = {
        "dataset_manifest.json",
        "partition_manifest.json",
        "human_audit.jsonl",
        "candidate_metrics.json",
        "candidate_rankings.npz",
        "bootstrap_receipt.json",
    }
    assert required == {path.name for path in output.iterdir()}
    assert all(path.stat().st_mode & 0o077 == 0 for path in output.iterdir())
    assert "raw query" not in "".join(
        path.read_text(encoding="utf-8") for path in output.glob("*.json")
    )


def test_structural_failure_writes_terminal_benchmark_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    (repository / "configs").mkdir(parents=True)
    (repository / "outputs").mkdir()
    config_path = repository / "configs" / CONFIG_PATH.name
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    preregistration = REPOSITORY_ROOT / "outputs" / "stage_c0_preregistration.md"
    (repository / "outputs" / preregistration.name).write_text(
        preregistration.read_text(encoding="utf-8"), encoding="utf-8"
    )
    labels = repository / "labels.jsonl"
    labels.write_text("{}\n" * 200, encoding="utf-8")

    monkeypatch.setattr(
        "qir_route.stage_c0.pipeline._git_path_is_frozen",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "qir_route.stage_c0.pipeline._git_implementation_lineage_frozen",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr("qir_route.stage_c0.pipeline._git_head", lambda root: "a" * 40)

    def reject_data(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise DataAcceptanceError("synthetic structural rejection")

    monkeypatch.setattr(
        "qir_route.stage_c0.pipeline._download_allowed_assets", reject_data
    )
    run_dir = run_stage_c0(config_path, labels)
    receipt = json.loads(
        (run_dir / "stage_c0_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["verdict"] == "benchmark_rejected"
    assert receipt["stage_c1_authorized"] is False
    assert receipt["gate_checks"]["structural_data_gates_passed"] is False
    assert len(receipt["artifact_sha256"]) == 9
    firewall = json.loads(
        (run_dir / "firewall_receipt.json").read_text(encoding="utf-8")
    )
    assert firewall["post_artifact_scan"]["firewall_intact"] is True


def test_rrf_metrics_and_lane_tie_break_are_deterministic() -> None:
    bm25 = np.asarray([[0, 1, 2], [1, 0, 2]], dtype=np.int32)
    dense = np.asarray([[1, 0, 2], [1, 2, 0]], dtype=np.int32)
    document_ids = ["d-0", "d-1", "d-2"]
    fused = reciprocal_rank_fusion(bm25, dense, document_ids, rrf_k=60, output_top_k=3)
    assert fused == [[0, 1, 2], [1, 0, 2]]
    metrics, vectors = evaluate_rankings(
        fused,
        document_ids=document_ids,
        gold_document_ids=["d-0", "d-1"],
    )
    assert metrics["recall_at_100"] == 1.0
    assert vectors["recall_at_100"].tolist() == [1.0, 1.0]
    tied = {lane: dict(metrics) for lane in ("bm25", "dense", "rrf")}
    assert choose_candidate_lane(tied) == "rrf"
    all_vectors = {lane: vectors for lane in ("bm25", "dense", "rrf")}
    intervals = paired_lane_intervals(
        all_vectors,
        selected_lane="rrf",
        replicates=10,
        confidence=0.95,
        seed=20260801,
    )
    assert set(intervals) == {"rrf_vs_dense", "rrf_vs_bm25", "dense_vs_bm25"}


def test_single_sample_bootstrap_is_deterministic() -> None:
    values = np.asarray([1, 1, 1, 0, 1, 0, 1, 1], dtype=np.float64)
    first = percentile_bootstrap_interval(
        values, replicates=500, confidence=0.95, seed=20260801
    )
    second = percentile_bootstrap_interval(
        values, replicates=500, confidence=0.95, seed=20260801
    )
    assert first == second
    assert first["point_estimate"] == 0.75


@pytest.mark.parametrize(
    ("structural", "human", "firewall", "complete", "recall", "lower", "expected"),
    [
        (False, True, True, True, 1.0, 1.0, "benchmark_rejected"),
        (True, False, True, True, 1.0, 1.0, "benchmark_rejected"),
        (True, True, False, True, 1.0, 1.0, "firewall_violation"),
        (True, True, True, False, 1.0, 1.0, "incomplete_technical_run"),
        (True, True, True, True, 0.8999, 1.0, "candidate_ceiling_inadequate"),
        (True, True, True, True, 0.90, 0.8799, "candidate_ceiling_inadequate"),
        (True, True, True, True, 0.90, 0.88, "stage_c1_authorized"),
    ],
)
def test_stage_c0_verdict_boundaries(
    structural: bool,
    human: bool,
    firewall: bool,
    complete: bool,
    recall: float,
    lower: float,
    expected: str,
) -> None:
    verdict, _ = stage_c0_verdict(
        structural_passed=structural,
        human_audit_passed=human,
        firewall_intact=firewall,
        complete=complete,
        recall_at_100=recall,
        bootstrap_lower=lower,
        minimum_recall=0.90,
        minimum_bootstrap_lower=0.88,
    )
    assert verdict == expected


def test_synthetic_smoke_exercises_three_lanes_without_data_access() -> None:
    receipt = run_synthetic_stage_c0_smoke()
    assert set(receipt["metrics"]) == {"bm25", "dense", "rrf"}
    assert receipt["network_accessed"] is False
    assert receipt["data_downloaded"] is False
    assert receipt["validation_rows_accessed"] is False
    assert receipt["test_rows_accessed"] is False
    assert receipt["smoke_passed"] is True
    assert receipt["scientific_verdict"] is None
    assert receipt["stage_c1_authorized"] is False
