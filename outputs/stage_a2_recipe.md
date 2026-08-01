# Stage A.2 Frozen Training Recipe

Status: frozen before the confirmation run.

## Data and provenance

- Dataset: CSConDa, 8,349 valid unique-context rows.
- Source: `upstream/ViRE/data/CSConDa.csv`.
- SHA-256: `52af2ceecf76c1db3300fb8373073e7d65b6c4fc53fdb2f23ad26d770f4cab4f`.
- License recorded by the project: CC-BY-NC-4.0.
- Context-group split: 70/15/15 with seed 20260731.
- Expected counts: 5,844 train, 1,252 validation, 1,253 sealed test.
- Test policy: assignment hashes only; no test embeddings, candidates, or metrics.

## Retrieval and model

- Encoder: `AITeamVN/Vietnamese_Embedding_v2` at revision `18b44161e041bf1d3a333ab5144b5b7b93f914d2`.
- Candidate retrieval: BM25 plus dense min-max fusion, dense weight 0.7, top 50.
- Missing gold candidates are injected only for training.
- Lanes: frozen baseline, QI-only, residual QI, matched residual classical.
- QI and classical heads each contain 6,912 parameters; residual lanes add one scalar weight.

## Optimization and evidence

- Seeds: 20260731, 20260732, 20260733, 20260734, 20260735.
- AdamW, learning rate 0.001, weight decay 0.0001.
- Multi-positive listwise loss, temperature 0.1.
- Batch size 32, at most 5 epochs, early-stopping patience 2.
- Gradient clipping: global norm 1.0, with clipped-step rate recorded.
- Metrics: Recall@50, MRR@10, nDCG@10, per-query exports without raw text.
- Paired bootstrap: 10,000 resamples, 95% confidence, seed 20260801.

## Promotion gate

- Residual QI is not below baseline in 5/5 seeds.
- Residual QI beats matched classical in at least 4/5 seeds.
- Mean paired delta nDCG@10 is positive and its 95% interval excludes zero.
- Validation has at least 200 queries and 50 candidates per query.
- Every learned lane has finite gradients.

Run with `uv run qir stage-a2-confirm --config configs/stage_a2_confirmation.yaml`.
