# QIR-Route/EViRAL Stage C.0 Research Decision Memo

**Decision ID:** `QIR-EVIRAL-C0-RDM-001`

**Controlling protocol:** `QIR-EVIRAL-C0-001`, revision 1

**Status:** `FROZEN — BENCHMARK REJECTED BEFORE HUMAN AUDIT`

**Effective date:** 2026-08-03 (Asia/Ho_Chi_Minh)

**Scope:** EViRAL data acceptance and candidate-ceiling preparation at Git
commit `c0c018a9bc9dec5492f6cfdb6a991a9eec960273`

**Technical evidence status:** verified

**Scientific disposition:** data-acceptance failure; candidate ceiling unassessed

## 1. Executive decision

The EViRAL Stage C.0 branch under decision `QIR-EVIRAL-C0-001` is closed with
verdict `benchmark_rejected`. The frozen internal partition places two distinct
normalized query values across both the fit and calibration partitions. These
two cross-partition values affect four queries: two fit queries and two
calibration queries.

This violates mandatory structural gate 9 of
`outputs/stage_c0_preregistration.md`. The frozen stopping rule prohibits
repairing rows, deduplicating queries, changing the partition, resplitting the
train set, or weakening the gate under the current decision ID. Human audit,
BM25, dense retrieval, RRF, bootstrap evaluation, generator selection, and
Stage C.1 authorization therefore did not run.

This result is a benchmark-and-partition acceptance failure. It is not evidence
that the EViRAL candidate ceiling is inadequate, that any candidate generator
fails, or that a quantum-inspired reranker is ineffective.

## 2. Research question and final answer

| ID | Frozen research question | Final answer |
|---|---|---|
| RQ-C0 | On a calibration partition frozen from official EViRAL train, does a fixed candidate generator achieve a sufficiently high candidate ceiling to authorize a quantum-inspired reranking study? | Not evaluated. A mandatory structural data gate failed before human audit or candidate generation, so the controlling verdict is `benchmark_rejected`. |

Consequently:

- `stage_c1_authorized` is false;
- no candidate lane was selected;
- no Recall, MRR, nDCG, bootstrap, latency, or memory result exists;
- official validation and test remain sealed;
- no scientific claim about QI-versus-classical reranking is authorized.

## 3. Frozen evidence chain

### 3.1 Source and implementation provenance

- Dataset: `NIRVLab/EViRAL`
- Dataset revision: `138308a5a1c647701b6f47bd7d14c919cd9c38fc`
- Recorded license: `CC-BY-4.0`
- Frozen implementation commit:
  `c0c018a9bc9dec5492f6cfdb6a991a9eec960273`
- Frozen implementation parent:
  `f49cdcb29255126ec628fc9efa5bf3446eeb358f`
- Remote `origin/main` matched the frozen implementation commit before data
  preparation.
- Stage C.0 preregistration SHA-256:
  `c3d97fa856aec529fe0dff34839c4952f2ea760d11f7e6915458b9e75651361e`

### 3.2 Allowed assets

Exactly the three preregistered assets were downloaded into an isolated cache.

| Asset | Bytes | Observed SHA-256 |
|---|---:|---|
| `corpus/corpus-00000-of-00001.parquet` | 27,095,806 | `210359d579fec0f2a45bcb358fa83a4ef48081f3040f89934d8bc610e5a978d4` |
| `queries/train-00000-of-00001.parquet` | 6,902,282 | `b536f8976901a9bd4136af853416ef14cb9b39e88209dabe71abe22254a3e334` |
| `qrels/train-00000-of-00001.parquet` | 1,200,401 | `b50110e0798d24140ae421bf2ea665f0dc2ac5cb9e004a7443c47c594fd50d1c` |

The observed row counts matched the protocol:

| Table | Rows |
|---|---:|
| Corpus | 123,972 |
| Train queries | 79,700 |
| Train qrels | 79,700 |

The query parquet schema contained `query_id`, `query_vi`, and `query_ede` as
upstream metadata. Only `query_id` and `query_vi` were materialized. No
validation or test query or qrel asset was downloaded, cached, or read.

### 3.3 Structural acceptance

The following gates passed before the controlling failure:

- frozen asset hashes and row counts matched;
- corpus IDs were non-empty and unique;
- train query IDs were non-empty and unique;
- train query and qrel query-ID sets matched;
- every query had exactly one positive qrel;
- every qrel corpus ID existed in the corpus;
- normalized queries and constructed documents were non-empty;
- fit and calibration ID sets were disjoint and had frozen counts.

The frozen partition produced:

- fit queries: `63,760`;
- calibration queries: `15,940`;
- calibration ordered-query-ID SHA-256:
  `51f6dd622fe651985ea1c40ab88a849b56d1db1ef9ce517898fde691547ab4c3`.

Mandatory structural gate 9 failed:

- cross-partition normalized duplicate values: `2`;
- affected queries: `4`;
- affected fit queries: `2`;
- affected calibration queries: `2`;
- aggregate hash of the normalized duplicate-value hashes:
  `c5e74ad0feeb02fb1a13a73c7d3dd5ff4adcafe281b86957c57540b365db2a87`.

No raw query or passage text is included in this memo or its rejection receipt.

### 3.4 Firewall and stopping evidence

The firewall passed before and after structural inspection:

- firewall intact: true;
- validation rows accessed: false;
- test rows accessed: false;
- forbidden artifacts discovered: none;
- unexpected cache files discovered: none.

The preparation stopped before audit sampling. Therefore:

- audit pairs selected: `0/200`;
- human audit started: false;
- retrieval outputs accessed: false;
- candidate generation run: false;
- model snapshot downloaded: false;
- raw text exported: false.

The controlling rejection receipt is stored locally at
`.git/stage_c0_audit/audit-prep-c0c018a9/prep_rejection_receipt.json`.
Its SHA-256 is
`e85ea2051938c79d16faf5aaeee563c9279747c515b5c977a027c18cceaca422`.

## 4. Decision ledger

### D1 — EViRAL Stage C.0 is benchmark-rejected

**Decision:** Close `QIR-EVIRAL-C0-001` with `benchmark_rejected`.

**Basis:** A normalized exact query occurs across fit and calibration for two
normalized values, violating mandatory structural gate 9.

**Consequence:** The data-acceptance failure controls the branch regardless of
any hypothetical candidate-retrieval performance.

### D2 — No rescue is permitted within the current decision ID

**Decision:** Do not deduplicate, drop rows, group duplicates, change the salt,
move queries between partitions, resplit official train, or weaken the gate
under `QIR-EVIRAL-C0-001`.

**Basis:** The behavior and the stopping rule were preregistered before the
allowed train assets were inspected.

**Consequence:** Any revised partition policy requires a new protocol revision
and a new decision ID before producing new scientific outputs.

### D3 — Human audit and candidate evaluation remain unrun

**Decision:** Do not create the 200-pair reviewer workflow or run BM25, dense,
RRF, bootstrap, or lane selection within this branch.

**Basis:** Structural acceptance is a mandatory predecessor of human audit and
candidate evaluation.

**Consequence:** No missing metric may be imputed, simulated, or described as a
Stage C.0 result.

### D4 — Stage C.1 is not authorized

**Decision:** Do not write or execute an EViRAL Stage C.1 confirmation protocol
as a continuation of `QIR-EVIRAL-C0-001`.

**Basis:** Stage C.1 requires every data-acceptance gate plus the candidate
Recall and bootstrap gates to pass. The first condition failed and the latter
conditions were never evaluated.

**Consequence:** No candidate generator, model revision, top-100 list, or lane
is promoted from this branch.

### D5 — Official validation and test remain sealed

**Decision:** Do not download, read, cache, embed, score, inspect, or materialize
official validation or test rows.

**Basis:** Stage C.0 did not pass, and no later promotion gate exists.

**Consequence:** The absence of validation and test results is a required
protocol outcome.

### D6 — Preserve the rejection evidence

**Decision:** Preserve the frozen commit, preregistration, three allowed asset
hashes, isolated cache, rejection receipt, and this memo as the controlling
audit trail.

**Basis:** These artifacts establish that the failure arose from a
preregistered structural gate rather than a download, schema, qrel, firewall,
or retrieval failure.

**Consequence:** Corrections must be additive. The controlling receipt and memo
must not be edited in place after they are frozen.

## 5. Permitted actions after closure

The following actions remain permitted:

- verify hashes, counts, schemas, firewall state, and the rejection receipt;
- document or publish the benchmark-acceptance failure without raw query text;
- remove the isolated cache later under a separately authorized cleanup action;
- design a new preregistration with a new decision ID;
- preregister a duplicate-group-aware partition rule before running a new data
  audit;
- reuse the shared corpus only under the firewall rules of a new protocol.

## 6. Prohibited actions within this branch

The following actions are prohibited:

- selecting or moving the four affected queries to make the current split pass;
- deduplicating after observing the failed gate;
- changing the normalization or partition salt under the current decision ID;
- choosing 200 audit pairs despite structural rejection;
- running candidate retrieval to see whether the rejected benchmark would have
  passed the ceiling gate;
- opening official validation or test;
- presenting the branch as evidence for or against a QI reranker;
- calling a revised experiment Stage C.0 under `QIR-EVIRAL-C0-001`.

## 7. Requirements for a genuinely new EViRAL program

A new EViRAL attempt is scientifically distinct and must be preregistered before
new evaluation. At minimum it must:

1. use a new decision ID and protocol revision;
2. state that `QIR-EVIRAL-C0-001` ended in `benchmark_rejected`;
3. freeze a deterministic duplicate-group-aware partition policy so normalized
   equivalents cannot cross fit and calibration;
4. freeze how group assignment interacts with the exact `15,940/63,760` target
   counts, including what happens when a group crosses the boundary;
5. retain the official validation/test firewall;
6. retain an isolated cache and receipt contract without raw query/passage text;
7. rerun all structural gates before human audit;
8. stop again without rescue if any newly preregistered mandatory gate fails.

The present memo does not select the new grouping algorithm, salt, thresholds,
or decision ID. Those choices belong in a separate preregistration and must be
fixed before the new protocol is run.

## 8. Final disposition

`QIR-EVIRAL-C0-001` is complete as a rejected benchmark-acceptance attempt.
There is no unfinished human audit, candidate evaluation, Stage C.1 run, or
validation/test evaluation to complete within this decision ID.

The branch may be cited only as evidence that the frozen query-level partition
was incompatible with its own cross-partition normalized-duplicate gate on the
frozen EViRAL train revision. All broader retrieval and QI questions remain
unanswered.
