# Stage C.0 Preregistration: EViRAL Data Acceptance and Candidate-Ceiling Gate

**Decision ID:** `QIR-EVIRAL-C0-001`

**Protocol revision:** 1

**Status:** `PREREGISTERED — NOT RUN`

**Effective date:** 2026-08-01 (Asia/Ho_Chi_Minh)

**Program:** New EViRAL research program; not a continuation or reopening of the
closed QIR-Route/CSConDa branch

## 1. Purpose and research boundary

Stage C.0 determines whether EViRAL is technically and scientifically acceptable
for a new QIR-Route study and whether a fixed, reproducible candidate generator
provides enough positive-document coverage to justify a later reranking study.

The frozen research question is:

> On a calibration partition frozen from the official EViRAL training split,
> does a fixed candidate generator achieve a sufficiently high candidate ceiling
> to authorize a quantum-inspired reranking study?

Vietnamese statement of the same question:

> Trên calibration partition được đóng băng từ official train, một candidate
> generator cố định có đạt ceiling đủ cao để cho phép nghiên cứu QI reranking
> hay không?

Stage C.0 is a benchmark-acceptance and candidate-ceiling stage. It must not
train, tune, score, or compare a quantum-inspired or classical reranker. It must
not access the official validation or test rows. A Stage C.0 pass authorizes only
the writing and preregistration of Stage C.1; it is not evidence that a QI method
is effective.

## 2. Prior-program separation

The QIR-Route/CSConDa branch is closed by
`outputs/research_decision_memo.md` under decision `QIR-CSCONDA-RDM-001`.
Stage C.0 creates a separate dataset-method program with a new decision ID,
dataset revision, partition rule, candidate protocol, artifacts, and gates.

The following are prohibited:

- presenting Stage C.0 as a rescue, Stage A.3, or Stage B.1 of CSConDa;
- reusing CSConDa validation or sealed-test observations for EViRAL decisions;
- modifying historical CSConDa configs, receipts, caches, checkpoints, or hashes;
- changing this protocol after observing any Stage C.0 scientific output without
  issuing a new protocol revision and decision ID.

## 3. Dataset contract

### 3.1 Source and schema

- Repository: [NIRVLab/EViRAL](https://huggingface.co/datasets/NIRVLab/EViRAL)
- Frozen revision: `138308a5a1c647701b6f47bd7d14c919cd9c38fc`
- License recorded by the dataset: CC BY 4.0
- Task used here: monolingual Vietnamese passage retrieval
- Query field: `query_vi`
- Prohibited query field: `query_ede`
- Document text: `title + "\n" + passage`; omit the title and newline when the
  stripped title is empty
- Relevance: binary; a qrel with a positive score identifies the labeled relevant
  corpus passage

The expected upstream sizes are:

| Config | Split | Expected rows |
|---|---|---:|
| `corpus` | corpus | 123,972 |
| `queries` | train | 79,700 |
| `qrels` | train | 79,700 |
| `queries` | validation | 17,078 |
| `qrels` | validation | 17,078 |
| `queries` | test | 17,079 |
| `qrels` | test | 17,079 |

The validation and test counts are provenance metadata only. Their row content
must not be retrieved during Stage C.0.

### 3.2 Allowed assets

Exactly these three data assets may be downloaded at the frozen revision:

| Asset | Frozen SHA-256 |
|---|---|
| `corpus/corpus-00000-of-00001.parquet` | `210359d579fec0f2a45bcb358fa83a4ef48081f3040f89934d8bc610e5a978d4` |
| `queries/train-00000-of-00001.parquet` | `b536f8976901a9bd4136af853416ef14cb9b39e88209dabe71abe22254a3e334` |
| `qrels/train-00000-of-00001.parquet` | `b50110e0798d24140ae421bf2ea665f0dc2ac5cb9e004a7443c47c594fd50d1c` |

The implementation must request these exact filenames rather than invoking a
loader that may materialize every split. It must use a run-isolated download
cache and record the observed SHA-256 of every allowed asset.

### 3.3 Validation and test firewall

The following assets, and any equivalent materialization of their rows, are
forbidden throughout Stage C.0:

- `queries/validation-00000-of-00001.parquet`
- `qrels/validation-00000-of-00001.parquet`
- `queries/test-00000-of-00001.parquet`
- `qrels/test-00000-of-00001.parquet`

Access to repository metadata, filenames, byte sizes, HEAD responses, and
dataset-card counts is allowed only when it does not return row content. The
shared corpus is allowed because it does not reveal validation/test relevance
without the sealed queries and qrels.

Before and after the run, the firewall verifier must scan the isolated cache and
run directory. Discovery of a forbidden file, forbidden row export, validation
or test candidate cache, or validation/test metric is a firewall failure. A
firewall failure cannot authorize Stage C.1 even if candidate metrics otherwise
pass.

## 4. Frozen internal partition

Only the 79,700 official training queries and their qrels may be partitioned.
For every train `query_id`, compute:

```text
SHA256("QIR-EVIRAL-C0-001\0" + query_id)
```

Sort query IDs by the 32-byte digest in ascending byte order, breaking an
impossible digest tie by ascending `query_id`. Assign the first 15,940 IDs to
`calibration` and the remaining 63,760 IDs to `fit`.

- Stage C.0 evaluates candidate retrieval only on `calibration`.
- Stage C.0 must not evaluate, aggregate, or report retrieval metrics on `fit`.
- Official validation remains sealed for Stage C.1 confirmation.
- Official test remains sealed until a later preregistered QI-versus-classical
  promotion gate passes.
- The partition manifest may contain IDs and hashes but no raw query or passage
  text.

## 5. Data-acceptance audit

All data gates are mandatory and precede candidate-generation evaluation.

### 5.1 Structural gates

The benchmark passes structural acceptance only if all of the following hold:

1. Repository revision, license, schemas, row counts, filenames, and asset hashes
   match this preregistration.
2. Every `corpus_id` is non-empty and unique.
3. Every train `query_id` is non-empty and unique.
4. The train query-ID set equals the train qrel query-ID set exactly.
5. Every query has exactly one positive qrel, and every qrel score is positive.
6. Every qrel `corpus_id` exists in the corpus.
7. Every selected `query_vi` and constructed document text is non-empty after
   normalization.
8. The fit and calibration ID sets are disjoint and have the frozen counts.
9. No normalized exact query string appears in both fit and calibration.

Normalization for gate 7 and gate 9 is Unicode NFKC, case folding, removal of
Unicode control and format characters, whitespace collapse, and stripping.
Positive passages may be shared across query partitions because the corpus is a
common retrieval collection.

Any structural-gate failure returns `benchmark_rejected`. The implementation
must not repair rows, deduplicate, resplit, or weaken the gate within this
decision ID.

### 5.2 Human relevance audit

The calibration queries are ordered independently within four query-length
quartiles. Query length is the number of tokens returned by the repository's
frozen Unicode token pattern. Quartile assignment is deterministic by ascending
length, with the partition digest from Section 4 as the tie-break. Each quartile
contains 3,985 queries.

Within each quartile, compute:

```text
SHA256("QIR-EVIRAL-C0-001-AUDIT\0" + query_id)
```

Select the 50 smallest digests per quartile, producing exactly 200 audit pairs.
One Vietnamese-fluent human reviewer must inspect `query_vi` and its labeled
positive passage without seeing any retriever output, lane name, rank, or metric.

The reviewer assigns exactly one label:

- `supported`: the passage contains enough information to answer or directly
  address the query;
- `not_supported`: the passage does not meet that criterion.

The reviewer also records zero or more fixed reason codes:
`non_vietnamese_query`, `empty_or_corrupt`, `topic_mismatch`,
`insufficient_answer`, `ambiguous`, or `other`. An ambiguous case is counted as
`not_supported`. The reviewer must not revise labels after candidate metrics are
revealed.

Human acceptance requires all three conditions:

- at least 180 of 200 pairs are labeled `supported`;
- at most 10 of 200 pairs carry `non_vietnamese_query`;
- zero pairs carry `empty_or_corrupt`.

The audit artifact contains only query ID, corpus ID, binary label, reason codes,
review timestamp, rubric version, and reviewer pseudonym. It must not export raw
query, title, or passage text.

## 6. Frozen candidate generators

No candidate method may use relevance feedback, gold-positive injection, encoder
training, reranker training, or hyperparameter tuning. Candidate metrics are
computed only after every data-acceptance gate passes.

### 6.1 BM25 lane

- Exact BM25 over all 123,972 constructed documents
- Existing repository Unicode tokenizer
- `k1 = 1.5`
- `b = 0.75`
- Retrieve top 1,000 documents per calibration query

### 6.2 Dense lane

- Encoder:
  [AITeamVN/Vietnamese_Embedding_v2](https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2)
- Frozen model revision: `18b44161e041bf1d3a333ab5144b5b7b93f914d2`
- Expected embedding dimension: 1,024
- Maximum sequence length: 2,048 tokens
- Query input: raw `query_vi`
- Document input: the frozen constructed document text
- L2-normalized float32 embeddings
- Exact dot-product retrieval; approximate-nearest-neighbor search is prohibited
- Retrieve top 1,000 documents per calibration query

The exact dense computation may be blockwise to bound memory, but block size
must not affect ranks. Corpus IDs ascending resolve exact score ties.

### 6.3 Reciprocal-rank-fusion lane

RRF uses only the frozen BM25 and dense top-1,000 rankings:

```text
RRF(d) = 1 / (60 + rank_bm25(d)) + 1 / (60 + rank_dense(d))
```

Ranks are one-based. A document absent from a base top-1,000 contributes zero
from that lane. RRF returns the top 100 documents. Exact RRF-score ties are
resolved by ascending `corpus_id`.

### 6.4 Lane selection

Compute calibration Recall@100 for BM25, dense, and RRF. Select the lane with the
highest Recall@100. Exact metric ties are resolved in this frozen order:

```text
RRF > dense > BM25
```

The selected candidate rule, preprocessing, model revision, ranks, top-k, and
tie-breaking policy become immutable inputs to a possible Stage C.1.

## 7. Metrics and uncertainty

The primary metric is calibration Recall@100. Because EViRAL has one labeled
positive per query, each per-query Recall@100 value is binary.

For the selected lane, draw 10,000 nonparametric bootstrap samples of the 15,940
per-query values with replacement. Use seed `20260801` and the percentile method
to report a two-sided 95% confidence interval for mean Recall@100.

Secondary diagnostics, which have no promotion authority, are:

- Recall@10, Recall@20, Recall@50, and Recall@100 for every lane;
- MRR@10 and nDCG@10 for every lane;
- paired per-query metric deltas among fixed lanes with 95% bootstrap intervals;
- indexing time, embedding time, query latency, peak memory, device, and software
  versions.

No correction, alternative confidence method, subgroup result, or secondary
metric may replace the primary decision rule.

## 8. Decision rule and stopping policy

Stage C.1 is authorized if and only if all conditions hold:

1. every structural data gate passes;
2. every human-audit gate passes;
3. the Stage C.0 firewall remains intact;
4. selected-lane Recall@100 is at least `0.90`;
5. the lower bound of its frozen 95% bootstrap interval is at least `0.88`.

The mutually exclusive terminal outcomes are:

| Observed condition | Frozen verdict | Consequence |
|---|---|---|
| Data or human-audit gate fails | `benchmark_rejected` | Stop; do not rescue, repair, or run candidate evaluation under this decision ID. |
| Data passes but Recall@100 is below 0.90 | `candidate_ceiling_inadequate` | Stop before QI training; secondary metrics cannot promote. |
| Recall@100 is at least 0.90 but CI lower bound is below 0.88 | `candidate_ceiling_inadequate` | Stop before QI training; point estimate alone is insufficient. |
| Firewall violation occurs | `firewall_violation` | Invalidate the run; Stage C.1 is not authorized. |
| Run terminates before every required artifact and gate exists | `incomplete_technical_run` | No scientific verdict; an identical-protocol rerun is allowed after an engineering fix. |
| Every mandatory gate passes | `stage_c1_authorized` | Freeze the selected generator and permit writing a separate Stage C.1 preregistration. |

Failure of a scientific gate closes this decision branch. It must not trigger a
new lane, tokenizer, encoder, fusion constant, top-k, sample, threshold, seed,
or audit reinterpretation. An engineering rerun is permitted only after
`incomplete_technical_run`, using the identical dataset revision, partition,
config, and scientific code path.

## 9. Required artifact contract

A complete future Stage C.0 run must produce receipt-backed artifacts below a
new run directory without raw text:

| Artifact | Required content |
|---|---|
| `preregistration_receipt.json` | Decision ID, protocol revision, preregistration SHA-256, Git commit SHA, config SHA-256 |
| `dataset_manifest.json` | Repository revision, license, schemas, counts, allowed filenames, observed asset hashes |
| `partition_manifest.json` | Salt, rule, fit/calibration counts, ordered ID hashes, overlap checks |
| `human_audit.jsonl` | IDs, labels, reason codes, timestamps, rubric revision, reviewer pseudonym |
| `firewall_receipt.json` | Allowed accesses, forbidden-path scan, validation/test access flags |
| `candidate_metrics.json` | Per-lane metrics, selected lane, deterministic tie result, promotion-gate observations |
| `candidate_rankings.npz` | Calibration query IDs and top document IDs/ranks only; no query or passage text |
| `bootstrap_receipt.json` | Seed, replicates, method, point estimate, interval, per-query-value hash |
| `cost_receipt.json` | Device, package versions, elapsed times, latency, peak memory |
| `stage_c0_receipt.json` | Integrity hashes, all gate booleans, final verdict, Stage C.1 authorization boolean |

Every artifact must be hashed from the final receipt. Missing artifacts imply
`incomplete_technical_run`, not a pass or failure of the scientific hypothesis.

## 10. Preregistration freeze and claim boundary

Before the first Stage C.0 run, this file must be committed. Its SHA-256, the
configuration SHA-256, and the full Git commit SHA must be recorded in the run's
preregistration receipt. The first scientific data access must occur only after
that freeze.

Any later change to dataset revision, allowed files, partitioning, audit rubric,
candidate lanes, tokenizer, encoder, fusion, metric, bootstrap procedure,
threshold, or stopping policy requires a new protocol revision and a new
decision ID before running. Documentation-only corrections must be explicitly
identified and must not change executable meaning.

A Stage C.0 pass supports only this claim:

> Under the frozen EViRAL training-calibration protocol, the selected fixed
> candidate generator achieved the preregistered candidate-recall gate and is
> adequate for a separately preregistered reranking study.

It does not support claims that QI reranking works, that QI beats a classical
control, that the result generalizes to official validation or test, or that the
closed CSConDa result has been reversed.
