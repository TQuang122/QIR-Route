# QIR-Route

Research code for **QIR-Route: Cost-Aware Quantum-Inspired Reranking for Vietnamese Information Retrieval**.

The repository is intentionally independent from the two upstream research repositories under `upstream/`. Those repositories are pinned, ignored by this Git repository, and used only for local reproduction and behavioral comparison because neither currently declares a software license.

## Stage 0 smoke baseline

```bash
uv sync --extra dev
uv run qir smoke-baseline --config configs/smoke_csconda.yaml
uv run pytest
```

The command evaluates BM25, the frozen Vietnamese embedding model, and ViRE-style min-max alpha fusion on a deterministic unique-context sample of CSConDa. It writes a manifest, metrics, per-query ranks, and timing receipt below `artifacts/stage0/`.

Stage 0 deliberately contains no quantum-inspired implementation. The QI head is introduced only after the baseline and provenance gates pass.

## Density-matrix math core

`qir_route.quantum` provides a batched nine-angle ZYZ reduction channel:

```python
from qir_route.quantum import density_matrix_reduce, squared_uhlmann_fidelity

rho_out = density_matrix_reduce(rho_a, rho_b, angles)
score = squared_uhlmann_fidelity(rho_out, target_density)
```

The reduction applies two local SU(2) rotations, a controlled SU(2), and a partial trace over the second qubit. Its output is a single-qubit density matrix and may be mixed.

The full shared head groups a normalized 1024-dimensional embedding into 256 four-scalar groups. Each group owns three independent nine-angle reduction nodes, giving exactly 6,912 trainable parameters:

```python
from qir_route.quantum import QuantumInspiredHead

head = QuantumInspiredHead()
scores = head.score(query_embeddings, top_50_document_embeddings, mode="mean")
assert head.quantum_parameter_count == 6912
```

Benchmark the paper-scale top-50 scoring surface and save a hardware receipt:

```bash
uv run qir benchmark-head --device auto --candidates 50 \
  --output artifacts/qi_head/top50_benchmark.json
```

## Stage A training smoke lane

The training smoke lane creates a deterministic context-group split, materializes only train and validation top-50 candidate caches, and trains the QI head with a multi-positive listwise objective:

```bash
uv run qir stage-a-smoke --config configs/stage_a_smoke.yaml
```

The test partition is assignment-only: no test candidate cache or test metric is produced before the screening configuration is frozen.

## Stage A.1 residual ablation

The residual lane starts exactly at the frozen hybrid baseline, injects missing
gold documents only into the training candidates, and leaves validation retrieval
unchanged. It compares QI-only, residual QI, and a symmetric classical group MLP
with the same 6,912 head parameters over three fixed seeds:

```bash
uv run qir stage-a1-ablation --config configs/stage_a1_ablation.yaml
```

The receipt reports Recall@50, MRR@10, nDCG@10, residual weight, gradient norms,
score variance, and a conservative promotion gate. The test partition remains
assignment-only throughout the ablation.

## Stage A.2 full-corpus confirmation

Stage A.2 uses all 8,349 unique CSConDa contexts, five fixed seeds, global-norm
gradient clipping, per-query metric receipts, and paired bootstrap confidence
intervals. Its test partition remains assignment-only:

```bash
uv run qir stage-a2-confirm --config configs/stage_a2_confirmation.yaml
```

The frozen recipe and promotion criteria are recorded in
`outputs/stage_a2_recipe.md`. A failed promotion gate is retained as a valid
negative result and does not trigger test evaluation.

## Post-Stage-A.2 diagnostics

The diagnostic namespace analyzes only frozen train/validation caches, receipts,
per-query exports, and checkpoints. It cannot tune, retrain, promote, or inspect
the sealed test split:

```bash
uv run qir verify-test-firewall \
  --stage-a2-run artifacts/stage_a2/20260801T094412Z-d5f23ea7
uv run qir provenance-snapshot \
  --config configs/post_a2_diagnostics.json \
  --output artifacts/post_a2_diagnostics/provenance_receipt.json
uv run qir diagnose-stage-a2 --config configs/post_a2_diagnostics.json
```

Historical Stage A receipts and artifacts remain immutable. Diagnostic outputs
are written below `artifacts/post_a2_diagnostics/` and explicitly state that they
cannot authorize a Stage A.3 run.
