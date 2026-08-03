# Stage C.0G Preregistration: EViRAL Group-Aware Data Acceptance and Candidate-Ceiling Gate

**Decision ID:** `QIR-EVIRAL-C0G-001`

**Protocol revision:** 1

**Status:** `PREREGISTERED — NOT RUN`

**Effective date:** 2026-08-03 (Asia/Ho_Chi_Minh)

**Program:** New group-aware EViRAL protocol; not a repair, continuation, or
reinterpretation of `QIR-EVIRAL-C0-001`

## 1. Purpose and research boundary

Stage C.0G determines whether EViRAL is technically and scientifically
acceptable for a new QIR-Route study and whether a fixed, reproducible candidate
generator provides enough positive-document coverage to justify a later
reranking study. It replaces query-level partition assignment with a
preregistered normalized-query-group assignment.

The frozen research question is:

> On a duplicate-group-aware calibration partition frozen from the official
> EViRAL training split, does a fixed candidate generator achieve a sufficiently
> high candidate ceiling to authorize a separately preregistered
> quantum-inspired reranking study?

Vietnamese statement of the same question:

> Trên calibration partition group-aware được đóng băng từ official EViRAL
> train, một candidate generator cố định có đạt ceiling đủ cao để cho phép một
> nghiên cứu QI reranking được preregister riêng hay không?

Stage C.0G is a benchmark-acceptance and candidate-ceiling stage. It must not
train, tune, score, or compare a quantum-inspired or classical reranker. It must
not access official validation or test rows. A pass authorizes only the writing
and preregistration of a separate Stage C.1G protocol; it is not evidence that a
QI method is effective.

No EViRAL data access, structural audit, human audit, candidate evaluation, or
model download has been performed under `QIR-EVIRAL-C0G-001` at the time this
document is written.

## 2. Separation from prior programs

### 2.1 Closed CSConDa program

The QIR-Route/CSConDa branch remains closed by
`outputs/research_decision_memo.md` under decision `QIR-CSCONDA-RDM-001`.
Stage C.0G must not reuse CSConDa validation or sealed-test observations, reopen
CSConDa stages, or modify historical CSConDa evidence.

### 2.2 Closed query-level EViRAL Stage C.0

`QIR-EVIRAL-C0-001` ended with `benchmark_rejected` before human audit because
its frozen query-level partition placed normalized-equivalent queries across
fit and calibration. The controlling closure artifact is
`outputs/stage_c0_research_decision_memo.md` under decision
`QIR-EVIRAL-C0-RDM-001`.

That prior outcome is immutable:

- its partition, receipt, memo, hashes, and verdict must not be rewritten;
- its missing human-audit and candidate metrics must not be imputed;
- its isolated cache is historical evidence, not the Stage C.0G run cache;
- Stage C.0G must not be described as a rerun, rescue, correction, or successful
  completion of `QIR-EVIRAL-C0-001`;
- a Stage C.0G outcome cannot change the prior `benchmark_rejected` verdict.

Stage C.0G is a new protocol motivated transparently by the known structural
failure. It uses a new decision ID, partition namespace, partition rule, cache,
output root, manifests, and receipts. No candidate result was observed under the
closed protocol.

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
- Relevance: binary; a qrel with a positive score identifies the single labeled
  relevant corpus passage

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

Validation and test counts are provenance metadata only. Their row content must
not be retrieved during Stage C.0G.

### 3.2 Allowed assets

Exactly these three data assets may be downloaded at the frozen revision:

| Asset | Frozen SHA-256 |
|---|---|
| `corpus/corpus-00000-of-00001.parquet` | `210359d579fec0f2a45bcb358fa83a4ef48081f3040f89934d8bc610e5a978d4` |
| `queries/train-00000-of-00001.parquet` | `b536f8976901a9bd4136af853416ef14cb9b39e88209dabe71abe22254a3e334` |
| `qrels/train-00000-of-00001.parquet` | `b50110e0798d24140ae421bf2ea665f0dc2ac5cb9e004a7443c47c594fd50d1c` |

The future implementation must request these exact filenames rather than use a
loader that may materialize every split. It must use a fresh run-isolated cache
under `cache/stage_c0g/<run_id>` and record the observed SHA-256 and byte count
of every allowed asset. It must not read or silently reuse the historical
`cache/stage_c0/audit-prep-c0c018a9` cache.

### 3.3 Validation and test firewall

The following assets, and any equivalent materialization of their rows, are
forbidden throughout Stage C.0G:

- `queries/validation-00000-of-00001.parquet`
- `qrels/validation-00000-of-00001.parquet`
- `queries/test-00000-of-00001.parquet`
- `qrels/test-00000-of-00001.parquet`

Repository metadata, filenames, byte sizes, HEAD responses, and dataset-card
counts may be accessed only when they contain no row data. The shared corpus is
allowed because it does not reveal validation/test relevance without the sealed
queries and qrels.

Before data loading, after data acceptance, after candidate evaluation, and
after final artifact creation, the firewall verifier must scan the isolated
cache and run directory. Discovery of a forbidden file, forbidden row export,
validation/test candidate cache, validation/test metric, symlink escape,
unexpected file, or unhashed cache file is a firewall failure.

## 4. Frozen text normalization

The following normalization is used for data gates and group construction:

1. convert the value to Unicode text;
2. apply Unicode NFKC normalization;
3. apply Unicode case folding;
4. remove every character whose Unicode category is `Cc` or `Cf`;
5. collapse each non-empty run of Unicode whitespace to one ASCII space;
6. strip leading and trailing whitespace.

This is the same normalization used by `QIR-EVIRAL-C0-001`. Empty normalized
queries are a structural failure and must be detected before group construction.
No stemming, accent removal, punctuation removal, translation, spelling repair,
or semantic clustering is allowed.

## 5. Frozen duplicate-group-aware partition

Only the 79,700 official training queries and their qrels may be partitioned.
All hashes below use UTF-8 input and return the 32 raw SHA-256 bytes for sorting;
hexadecimal notation is used only for manifests and the formulas below.

### 5.1 Group construction

For each train query, compute `normalized_query` using Section 4. Queries belong
to the same group if and only if their complete normalized strings are equal.
For each distinct group, compute:

```text
group_key = SHA256(normalized_query).hexdigest()
```

Group membership is determined by exact normalized-string equality, not by hash
equality. If two distinct normalized strings produce the same `group_key`, the
run returns `benchmark_rejected`; the implementation must not merge or manually
disambiguate the colliding groups.

Every query ID must occur in exactly one group. Groups must be non-empty, and
the sum of all group sizes must equal 79,700.

### 5.2 Group ordering

For every group, compute:

```text
group_order_key =
  SHA256("QIR-EVIRAL-C0G-001\0" + group_key)
```

Here `group_key` is its 64-character lowercase hexadecimal representation, and
`\0` is one zero byte. Sort groups by the 32 raw bytes of `group_order_key` in
ascending order. Break an impossible order-key tie by ascending raw bytes of
`group_key`.

The group order is a partition input only. It must not depend on qrel corpus ID,
query length, raw query ordering, candidate results, or manual choices.

### 5.3 Prefix-boundary selection

Let the ordered groups be `g_1, ..., g_G`, let `s_i` be the number of queries in
`g_i`, and define the cumulative query count:

```text
C_m = sum(s_i for i = 1..m)
```

Consider every prefix boundary `m` in `1..G-1`, so both partitions remain
non-empty. Select the unique boundary that minimizes this tuple
lexicographically:

```text
(
  abs(C_m - 15940),
  C_m,
  group_order_key(g_m),
  group_key(g_m)
)
```

This implements the frozen tie rules:

1. choose the prefix whose total query count is closest to 15,940;
2. if two prefix counts are equally close, choose the smaller count;
3. if still tied, choose the smaller boundary `group_order_key`;
4. if still tied, choose the smaller boundary `group_key`.

No group may be split to reach 15,940 exactly. Calibration contains every query
in groups `g_1..g_m`; fit contains every query in the remaining groups.

### 5.4 Actual-count freeze

The deterministic boundary produces `actual_calibration_count = C_m` and
`actual_fit_count = 79700 - C_m`. These observed counts are outputs of the
preregistered algorithm, not analyst choices.

Immediately after successful partition construction, the runner must record and
freeze:

- total group count;
- singleton and non-singleton group counts;
- maximum group size;
- selected boundary index and its two hashes;
- target calibration count: 15,940;
- actual calibration and fit counts;
- absolute target deviation;
- ordered group-key SHA-256;
- ordered calibration-query-ID SHA-256;
- ordered fit-query-ID SHA-256.

The actual counts become immutable inputs to the remaining Stage C.0G run. They
must not be edited, rounded, resampled, or used to choose a second boundary.

### 5.5 Partition use restrictions

- Candidate retrieval is evaluated only on actual calibration queries.
- Fit queries must not be evaluated, aggregated, or reported in Stage C.0G
  retrieval metrics.
- No normalized group may cross fit and calibration.
- Official validation remains sealed for a possible Stage C.1G confirmation.
- Official test remains sealed until a later preregistered QI-versus-classical
  promotion gate passes.
- Partition artifacts may contain IDs and cryptographic hashes but no raw query,
  title, or passage text.

## 6. Data-acceptance audit

All data gates are mandatory and precede human audit and candidate evaluation.

### 6.1 Structural gates

The benchmark passes structural acceptance if and only if all conditions hold:

1. Repository revision, license, schemas, filenames, row counts, asset byte
   counts, and asset SHA-256 values match this preregistration.
2. Every `corpus_id` is non-null, non-empty, and unique.
3. Every train `query_id` is non-null, non-empty, and unique.
4. The train query-ID set equals the train qrel query-ID set exactly.
5. Every query has exactly one qrel, and every qrel score is positive.
6. Every qrel `corpus_id` exists in the corpus.
7. Every `query_vi` and constructed document text is non-empty after frozen
   normalization.
8. `query_ede` is not materialized or exported.
9. Exact normalized-string groups cover all 79,700 queries exactly once.
10. No two distinct normalized strings share a `group_key`.
11. Group order and the selected prefix boundary equal the frozen algorithm in
    Section 5 exactly.
12. Fit and calibration are non-empty, disjoint, exhaustive, and their counts
    equal the recorded actual counts.
13. No normalized group has members in both fit and calibration.
14. Every one of the four calibration query-length quartiles contains at least
    50 queries.
15. Firewall scans before and after data acceptance remain intact.

Positive passages may be shared across partitions because the corpus is a
common retrieval collection. Query groups must not be reassigned based on their
qrel or positive passage.

Any structural-gate failure returns `benchmark_rejected`. The implementation
must not repair rows, deduplicate, merge hash collisions, split groups, select a
different prefix, change the target, resplit, or weaken a gate within this
decision ID.

### 6.2 Human relevance audit

After structural acceptance, order actual calibration queries by ascending
query length. Query length is the token count from the repository's frozen
Unicode tokenizer. For equal lengths, compute:

```text
quartile_tie_key =
  SHA256("QIR-EVIRAL-C0G-001\0" + query_id)
```

Break length ties by ascending raw `quartile_tie_key`, then ascending
`query_id`. For zero-indexed ordered position `i` among `N` actual calibration
queries, assign:

```text
quartile(i) = min(floor(4 * i / N), 3)
```

Record and freeze the four resulting quartile counts. Within each quartile,
compute:

```text
SHA256("QIR-EVIRAL-C0G-001-AUDIT\0" + query_id)
```

Select the 50 smallest digests per quartile, breaking an impossible digest tie
by ascending `query_id`. This produces exactly 200 audit pairs.

One Vietnamese-fluent human reviewer must inspect `query_vi` and its labeled
positive passage without seeing any retriever output, lane name, rank, metric,
promotion threshold, or aggregate audit result.

The reviewer assigns exactly one label:

- `supported`: the passage contains enough information to answer or directly
  address the query;
- `not_supported`: the passage does not meet that criterion.

The reviewer records zero or more fixed reason codes:
`non_vietnamese_query`, `empty_or_corrupt`, `topic_mismatch`,
`insufficient_answer`, `ambiguous`, or `other`. An ambiguous pair must be
`not_supported`. Labels must not be revised after retrieval outputs are
revealed.

Human acceptance requires all conditions:

- at least 180 of 200 pairs are `supported`;
- at most 10 of 200 pairs carry `non_vietnamese_query`;
- zero pairs carry `empty_or_corrupt`;
- exactly one reviewer pseudonym occurs across all 200 records;
- every selected query/corpus pair matches the frozen qrel.

The audit artifact contains only query ID, corpus ID, label, reason codes,
review timestamp, rubric version, and reviewer pseudonym. It must not export raw
query, title, passage, constructed document text, retrieval output, rank, or
metric.

## 7. Frozen candidate generators

Candidate metrics are computed only after every structural and human-audit gate
passes. Relevance feedback, positive injection, query expansion, encoder
training, reranker training, and hyperparameter tuning are prohibited.

### 7.1 BM25 lane

- Exact BM25 over all 123,972 constructed documents
- Existing repository Unicode tokenizer
- `k1 = 1.5`
- `b = 0.75`
- Retrieve top 1,000 documents per actual calibration query

### 7.2 Dense lane

- Encoder:
  [AITeamVN/Vietnamese_Embedding_v2](https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2)
- Frozen model revision: `18b44161e041bf1d3a333ab5144b5b7b93f914d2`
- Expected embedding dimension: 1,024
- Maximum sequence length: 2,048 tokens
- Query input: raw `query_vi` in memory only
- Document input: frozen constructed document text in memory only
- L2-normalized float32 embeddings
- Exact normalized dot-product retrieval
- Approximate-nearest-neighbor search is prohibited
- Retrieve top 1,000 documents per actual calibration query

Exact dense computation may be blockwise to bound memory, but block size must
not affect ranks. Ascending `corpus_id` resolves exact score ties.

### 7.3 Reciprocal-rank-fusion lane

RRF uses only the frozen BM25 and dense top-1,000 rankings:

```text
RRF(d) = 1 / (60 + rank_bm25(d)) + 1 / (60 + rank_dense(d))
```

Ranks are one-based. A document absent from one base top-1,000 contributes zero
from that lane. RRF returns top 100. Ascending `corpus_id` resolves exact
RRF-score ties.

### 7.4 Lane selection

Compute actual-calibration Recall@100 for BM25, dense, and RRF. Select the lane
with the highest Recall@100. Exact metric ties are resolved in this order:

```text
RRF > dense > BM25
```

The selected preprocessing, generator, model revision, top-100 results, ranks,
and tie rules become immutable inputs to a possible Stage C.1G.

## 8. Metrics and uncertainty

The primary metric is actual-calibration Recall@100. Because EViRAL has one
labeled positive per query, each per-query Recall@100 value is binary.

For the selected lane, draw 10,000 nonparametric bootstrap samples with
replacement from the complete vector of `actual_calibration_count` per-query
binary values. Use seed `20260801` and the percentile method to report a
two-sided 95% confidence interval for mean Recall@100.

Secondary diagnostics have no promotion authority:

- Recall@10, Recall@20, Recall@50, and Recall@100 for every lane;
- MRR@10 and nDCG@10 for every lane;
- paired per-query deltas among all three fixed lanes with 95% bootstrap
  intervals;
- indexing time, embedding time, exact-search time, RRF time, query latency,
  peak memory, device, and software versions.

No alternative interval, correction, subgroup, group-weighted metric, fit-set
metric, or secondary metric may replace the primary decision rule. Each query,
including queries in a non-singleton normalized group, receives one equal unit
of weight in the frozen metrics and bootstrap.

## 9. Decision rule and stopping policy

Stage C.1G is authorized if and only if all conditions hold:

1. every structural data gate passes;
2. every human-audit gate passes;
3. the Stage C.0G firewall remains intact through final artifact creation;
4. selected-lane Recall@100 is at least `0.90`;
5. the lower bound of its frozen 95% bootstrap interval is at least `0.88`.

The mutually exclusive terminal outcomes are:

| Observed condition | Frozen verdict | Consequence |
|---|---|---|
| Asset, schema, ID, qrel, text, grouping, boundary, count, quartile, or human-audit gate fails | `benchmark_rejected` | Stop; do not repair, resplit, regroup, resample, or run candidate evaluation under this decision ID. |
| Data and audit pass, but selected Recall@100 is below 0.90 | `candidate_ceiling_inadequate` | Stop before QI training; secondary metrics cannot promote. |
| Recall@100 is at least 0.90, but its CI lower bound is below 0.88 | `candidate_ceiling_inadequate` | Stop before QI training; the point estimate alone is insufficient. |
| A firewall violation occurs at any time | `firewall_violation` | Invalidate the run; Stage C.1G is not authorized. |
| The run terminates before every required artifact and gate exists | `incomplete_technical_run` | No scientific verdict; rerun only after an engineering fix with identical scientific inputs. |
| Every mandatory gate passes | `stage_c1g_authorized` | Freeze the selected generator and permit a separate Stage C.1G preregistration. |

Failure of a scientific gate closes this decision branch. It must not trigger a
new partition boundary, normalization, lane, tokenizer, encoder, fusion
constant, top-k, audit sample, reviewer reinterpretation, threshold, seed, or
bootstrap method.

An engineering rerun is permitted only after `incomplete_technical_run`, using
the identical dataset revision, fresh isolated paths, code commit, config,
grouping algorithm, selected boundary, actual counts, audit sample, candidate
rules, and scientific gates. `benchmark_rejected`,
`candidate_ceiling_inadequate`, and `firewall_violation` cannot be converted
into technical reruns.

## 10. Prohibited actions

Within `QIR-EVIRAL-C0G-001`, it is prohibited to:

- edit or reinterpret the closed `QIR-EVIRAL-C0-001` evidence;
- read the historical raw cache as the Stage C.0G run input;
- download, read, cache, or materialize official validation/test rows;
- materialize `query_ede` beyond allowed schema metadata inspection;
- change normalization after seeing group sizes or the selected boundary;
- merge distinct normalized values, split a normalized group, move individual
  queries, or choose a non-minimal prefix boundary;
- change the target 15,940 after partition construction;
- discard duplicate queries or reweight them after seeing metrics;
- change quartile or audit sampling after seeing sampled pairs;
- expose retrieval outputs to the human reviewer;
- revise human labels after retrieval output exists;
- add a candidate lane after seeing BM25, dense, or RRF results;
- use ANN, relevance feedback, gold-positive injection, or encoder tuning;
- train or compare QI/classical rerankers in Stage C.0G;
- promote on a secondary metric or an unregistered subgroup;
- open Stage C.1G or official validation unless every mandatory gate passes;
- open official test without a later preregistered QI-versus-classical
  promotion gate.

## 11. Required artifact and privacy contract

A complete future run must use `artifacts/stage_c0g/<run_id>` and produce the
following receipt-backed artifacts without raw query or passage text:

| Artifact | Required content |
|---|---|
| `preregistration_receipt.json` | Decision ID, protocol revision, preregistration SHA-256, config SHA-256, full Git commit SHA |
| `dataset_manifest.json` | Repository revision, license, schemas, counts, allowed filenames, observed byte counts and asset hashes |
| `group_manifest.json` | Normalization revision, group counts/sizes, collision check, ordered group-key hash; no normalized strings |
| `partition_manifest.json` | Target, boundary tuple, actual counts, group-integrity checks, ordered ID hashes, fit/calibration overlap checks |
| `human_audit.jsonl` | IDs, labels, reason codes, timestamps, rubric revision, reviewer pseudonym |
| `firewall_receipt.json` | Allowed accesses, cache/output allowlists and hashes, forbidden-path scans, validation/test access flags |
| `candidate_metrics.json` | Per-lane metrics, selected lane, deterministic tie result, promotion observations |
| `candidate_rankings.npz` | SHA-256 identifiers for calibration queries and top documents only; no raw IDs or text |
| `bootstrap_receipt.json` | Seed, replicates, method, point estimate, interval, query count, per-query-value hash |
| `cost_receipt.json` | Device, package versions, elapsed times, latency, peak memory, model source hashes |
| `stage_c0g_receipt.json` | Final artifact hashes, all gate booleans, terminal verdict, Stage C.1G authorization boolean |

Every JSON/JSONL artifact must have a strict filename-specific schema and typed
values. Unknown keys, unexpected files, raw exception messages, unhashed model
cache files, or non-hashed candidate-ranking identifiers are firewall failures.

The final firewall scan must occur after the terminal receipt exists. Human
review may display raw query and positive passage transiently to the reviewer,
but the reviewer interface must read them directly from verified train assets
and must not persist them in audit, cache-derived, log, crash, or run artifacts.

Missing required artifacts imply `incomplete_technical_run`, never a scientific
pass or failure.

## 12. Verification scenarios

Before freezing implementation, automated tests must cover at least:

1. exact group construction and group-key collision rejection;
2. deterministic group order independent of input row order;
3. prefix-boundary selection with exact target, equal-distance target, and
   impossible-exact-target cases;
4. the smaller-count and boundary-hash tie rules;
5. group-integrity failure across fit/calibration;
6. actual-count and quartile-count freezing;
7. exactly 50 audit samples per quartile and 200 total;
8. data failure, human-audit failure, point-recall failure, CI failure,
   firewall failure, technical interruption, and full pass;
9. rejection of validation/test paths, raw JSON values, raw NPZ values,
   unexpected files, symlinks, and additional cache files without hashes;
10. dry-run and synthetic smoke with no network, data download, scientific
    verdict, or Stage C.1G authorization.

## 13. Preregistration freeze and claim boundary

Before the first Stage C.0G data access, this document must be committed and its
SHA-256 recorded. The Stage C.0G configuration, implementation, tests, and
scientific dependencies must then be committed and frozen. The first
`QIR-EVIRAL-C0G-001` structural audit may begin only when a dry-run verifies:

- preregistration hash and Git freeze;
- configuration hash and Git freeze;
- exact implementation lineage and clean scientific paths;
- fresh `cache/stage_c0g` and `artifacts/stage_c0g` run paths;
- validation/test firewall readiness;
- no candidate or human-audit output from an earlier Stage C.0G attempt.

Any change to dataset revision, allowed files, normalization, grouping,
partition boundary, audit rubric, candidate lanes, tokenizer, encoder, fusion,
metric, bootstrap, threshold, privacy schema, or stopping rule requires a new
protocol revision and a new decision ID before data access. Documentation-only
corrections must be identified explicitly and must not change executable
meaning.

A Stage C.0G pass supports only this claim:

> Under the frozen group-aware EViRAL training-calibration protocol, the
> selected fixed candidate generator achieved the preregistered candidate-recall
> gate and is adequate for a separately preregistered reranking study.

It does not support claims that QI reranking works, that QI beats a classical
control, that the result generalizes to official validation or test, that
duplicate grouping improves retrieval, or that the closed CSConDa or
`QIR-EVIRAL-C0-001` outcomes have been reversed.
