# QIR-Route/CSConDa Research Decision Memo

**Decision ID:** `QIR-CSCONDA-RDM-001`

**Status:** `FROZEN — RESEARCH BRANCH CLOSED`

**Effective date:** 2026-08-01 (Asia/Ho_Chi_Minh)

**Scope:** QIR-Route experiments using the frozen CSConDa split and artifacts

**Technical evidence status:** verified

**Scientific disposition:** negative but informative result

## 1. Executive decision

The QIR-Route/CSConDa research branch is closed. The frozen quantum-inspired
(QI) reranker is not promoted, Stage A.3 is not justified, and the Stage B.0
candidate-ceiling audit does not justify Stage B.1. The sealed CSConDa test
partition remains unopened and is not authorized for evaluation.

No additional tuning, retraining, candidate strategy, validation-driven rescue,
or threshold revision may be presented as a continuation of this experimental
line. Reproduction, auditing, documentation, packaging, and publication of the
negative result remain allowed as long as they do not create new scientific
claims from the sealed test partition or modify frozen evidence.

This memo closes a method-and-dataset pairing. It does not establish that every
quantum-inspired retrieval method is ineffective, nor that the current method
must fail on a different benchmark or under a separately preregistered protocol.

## 2. Research questions and final answers

| ID | Frozen research question | Final answer |
|---|---|---|
| RQ1 | Can the 6,912-parameter QI head outperform the frozen fused retrieval baseline on CSConDa? | No evidence of improvement under the full-corpus confirmation protocol. |
| RQ2 | Does residual QI outperform a parameter-matched residual classical head reliably? | No. Residual QI beat the matched classical lane in only 1/5 Stage A.2 seeds. |
| RQ3 | Is there a stable validation slice that justifies a new Stage A.3? | No. Post-A.2 diagnostics found no stable QI-helpful regime. |
| RQ4 | Is candidate absence the principal recoverable cause of the failed reranker result? | Only partially. Fixed dense augmentation recovered some missing positives, but failed the preregistered effect-size and recovery gates. |
| RQ5 | Is a separately preregistered Stage B.1 candidate-generation study justified on this branch? | No. Stage B.0 returned `candidate_ceiling_not_recoverable`. |
| RQ6 | Should the sealed CSConDa test partition now be opened? | No. No promotion gate passed, so test evaluation has no scientific authorization. |

## 3. Frozen evidence chain

### 3.1 Stage A smoke

Stage A established that the training, candidate-cache, QI-head, receipt, and
test-firewall surfaces executed correctly on a 128-query smoke corpus. It was a
technical viability check, not promotion evidence.

- Baseline validation nDCG@10: `0.835473`
- Best QI validation nDCG@10: `0.792663`
- Validation support: 19 queries
- Test firewall intact: true
- Receipt SHA-256:
  `7aea138f63e801cad50e0811eae0983c31976f743932e9bca370437cdb29b742`

### 3.2 Stage A.1 ablation

The three-seed smoke ablation produced an encouraging residual-QI signal on the
same small validation set, but failed its frozen comparator gate.

- Baseline mean nDCG@10: `0.835473`
- Residual-QI mean nDCG@10: `0.863402`
- Residual-classical mean nDCG@10: `0.859861`
- Residual QI not below baseline: 3/3 seeds
- Residual QI beat matched classical: 2/3 seeds
- Frozen promotion verdict: `promotion_rejected`
- Receipt SHA-256:
  `e23f991796d422e89df024916ff33e9d54bc9a3d282323bf155ef2048ddaa2aa`

The Stage A.1 result is retained as a small-sample screening observation. It is
not treated as confirmation because it used only 19 validation queries and did
not pass the preregistered comparator gate.

### 3.3 Stage A.2 full-corpus confirmation

Stage A.2 evaluated five fixed seeds on 1,252 validation queries with the frozen
baseline, QI-only, residual-QI, and matched residual-classical lanes.

| Lane | Mean validation nDCG@10 |
|---|---:|
| Frozen fused baseline | 0.448165 |
| QI only | 0.386919 |
| Residual QI | 0.431202 |
| Residual classical | 0.438160 |

Frozen gate observations:

- Residual QI not below baseline: 0/5 seeds; required 5/5.
- Residual QI beat matched classical: 1/5 seeds; required at least 4/5.
- Mean paired residual-QI delta versus baseline: not positive.
- Paired nDCG@10 confidence interval excluded zero in the positive direction:
  false.
- Gradients finite: true.
- Scientific status: `promotion_rejected`.
- Natural validation Recall@50: `0.714856`.
- Receipt SHA-256:
  `61f63ce24b42e10dba537a42b46cb23efba745b1362ecf17ae8c0ed5be3e38c3`

The small Stage A.1 signal therefore did not replicate at the confirmation
scale. Stage A.2 is the controlling evidence for the QI promotion decision.

### 3.4 Post-A.2 diagnostics

Diagnostic-only analysis preserved all frozen receipts and evaluated whether a
stable validation regime could explain or localize QI benefit.

- Diagnostic verdict: `insufficient_evidence`
- Stable QI-helpful regime found: false
- Strongest valid slice: none
- Stage A.3 scientifically justified: false
- Test remained untouched: true
- Diagnostic receipt SHA-256:
  `009b78b4a6f8c2d310045b2aaea760d5c9341588154c47bf54b24a2129dd22d9`

This result closes validation-slice rescue of the frozen QI method. A slice may
be described diagnostically, but no observed slice authorizes a new QI stage.

### 3.5 Stage B.0 candidate-ceiling recovery audit

The audit reconstructed a full dense validation score matrix solely from stored
normalized validation embeddings. It evaluated exactly five fixed strategies:

1. `frozen_fused_top50`
2. `dense_top50`
3. `dense_top100`
4. `fused50_union_dense50`
5. `fused50_union_dense100`

The best fixed strategy was `fused50_union_dense100`.

- Frozen candidate Recall: `0.714856`
- Best candidate Recall: `0.755591`
- Absolute improvement: `+0.040735`
- Paired bootstrap 95% CI: `[0.030351, 0.051917]`
- Bootstrap replicates: 10,000
- Bootstrap seed: 20260801
- Missing positive documents under frozen Top-50: 357/1,252
- Missing queries recovered: 51/357 (`14.286%`)
- Already-retrievable queries lost by stable union: 0/895
- Candidate-ceiling verdict: `candidate_ceiling_not_recoverable`
- Stage B.1 justified: false
- Stage A.3 justified: false
- Candidate-ceiling receipt SHA-256:
  `5100160a8064b7e500d55ba1b83d87ffc2fcf4c7a1ad27d6216df255dee18e0c`

The improvement was statistically positive, but it failed both mandatory
practical gates: absolute Recall improvement was below `0.05`, and missing-query
recovery was below `20%`. Statistical significance alone is not sufficient for
promotion under the frozen decision rule.

## 4. Decision ledger

### D1 — Frozen QI promotion is rejected

**Decision:** Final for the QIR-Route/CSConDa branch.

**Basis:** Stage A.2 failed every comparative performance requirement despite
finite gradients and adequate validation support.

**Consequence:** The frozen QI method must not be described as improving the
baseline or the matched classical control.

### D2 — Stage A.3 is closed

**Decision:** Do not implement or run Stage A.3.

**Basis:** Stage A.2 rejected promotion and post-A.2 diagnostics found no stable,
adequately supported QI-helpful regime.

**Consequence:** No validation-slice, seed subset, checkpoint, or alternative
summary may be used to reopen Stage A.3 within this branch.

### D3 — Candidate-ceiling rescue is rejected

**Decision:** Do not implement or run Stage B.1 on the frozen branch.

**Basis:** The best fixed union improved candidate Recall by `0.040735` and
recovered `14.286%` of missing queries, below the frozen `0.05` and `20%` gates.

**Consequence:** The positive bootstrap interval may be reported as a diagnostic
signal, but not as evidence that the candidate ceiling is recoverable enough to
justify a new candidate-generation method.

### D4 — CSConDa test remains sealed

**Decision:** Do not read, embed, cache, score, inspect, or materialize the test
partition.

**Basis:** No scientific promotion gate passed.

**Consequence:** There will be no final test number for this closed method branch.
The absence of a test result is a protocol outcome, not missing experimental
work.

### D5 — Frozen evidence remains immutable

**Decision:** Preserve historical configs, receipts, checkpoints, caches,
manifests, per-query exports, and run artifacts byte-for-byte.

**Basis:** Their hashes define the audit trail supporting this memo.

**Consequence:** Corrections or reinterpretations must be written as new,
diagnostic-only artifacts; historical files must not be edited in place.

### D6 — Retain the work as a negative result

**Decision:** Preserve and report the complete failure chain rather than tuning
until a positive result appears.

**Basis:** Stage A.1's small-sample signal did not replicate, matched classical
controls outperformed residual QI at confirmation scale, and candidate expansion
was insufficient under frozen gates.

**Consequence:** Suitable future work includes reproducibility documentation,
an audit or negative-results manuscript, and a new independently preregistered
study. It does not include further CSConDa validation rescue.

## 5. Actions permitted after closure

The following actions remain in scope:

- Re-run deterministic verifiers against existing artifacts.
- Improve tests, documentation, packaging, and receipt readers without changing
  scientific outputs.
- Reproduce an existing frozen run using the exact frozen configuration and
  label the result as a reproduction.
- Write a technical report, audit paper, or negative-results manuscript using
  the frozen evidence chain.
- Design a new study with a new decision ID, new preregistration, and preferably
  a benchmark with official train/development/test partitions.
- Compare a new QI hypothesis against a parameter-, input-, and compute-matched
  classical control only after candidate recall is demonstrated to be adequate
  under the new protocol.

## 6. Actions prohibited within this branch

The following actions would invalidate the closure boundary:

- Opening or evaluating the sealed CSConDa test partition.
- Changing promotion thresholds after observing Stage A.2 or B.0 results.
- Adding candidate strategies to Stage B.0 after seeing its fixed-strategy
  outcomes.
- Selecting favorable seeds, checkpoints, queries, or slices as confirmation.
- Retraining the frozen QI or classical heads and calling the result Stage A.2,
  Stage A.3, B.0, or B.1.
- Treating a positive bootstrap interval as sufficient when an effect-size gate
  fails.
- Claiming that `fused50_union_dense100` is a promoted retrieval method.
- Claiming that the closed result disproves quantum-inspired retrieval in
  general.
- Modifying historical artifacts and regenerating hashes to conceal the change.

## 7. Conditions for a genuinely new research program

Future QI retrieval work must not be represented as reopening this branch. A new
program requires all of the following before model training:

1. A new decision and experiment identifier.
2. A preregistered research question and falsifiable primary hypothesis.
3. Official or independently frozen train/development/test partitions.
4. A sealed test policy established before development.
5. A strong, preregistered candidate-generation baseline with an explicit
   validation candidate-recall gate.
6. A parameter-, feature-, and compute-matched classical comparator.
7. Fixed seeds, metrics, confidence procedure, minimum support, and stopping
   rules.
8. A rule that failure of any promotion gate closes the new branch without
   post-hoc rescue.

The recommended new research question is narrower than the original claim:

> When the candidate ceiling is demonstrably adequate, does a quantum-inspired
> reranker provide reproducible improvement over a matched classical reranker?

This question must be tested on a separately governed protocol. The current
CSConDa validation and sealed test partitions must not be reused as a fresh
development surface.

## 8. Claim boundary for reports and papers

### Claims supported by the evidence

- The implementation and research protocol executed successfully.
- The frozen QI method failed the full-corpus CSConDa promotion gate.
- Residual QI did not reliably outperform the matched classical control.
- No adequately supported stable QI-helpful validation regime was found.
- Fixed dense candidate augmentation produced a positive but insufficient
  candidate-recall improvement.
- The sealed test partition remained untouched throughout the evidence chain.

### Claims not supported by the evidence

- Quantum-inspired retrieval methods are universally ineffective.
- The QI head cannot work on any other dataset or candidate distribution.
- Candidate generation can never improve QIR-Route.
- Stage B.0's best union improves top-10 end-to-end ranking quality.
- A hidden favorable test result exists or would reverse the decision.
- Failure is attributable to a single causal mechanism.

## 9. Evidence registry

| Artifact | Role | Frozen SHA-256 or status |
|---|---|---|
| `outputs/stage_a2_recipe.md` | Preregistered Stage A.2 recipe | frozen before confirmation |
| `artifacts/stage_a/20260801T052953Z-51890358/stage_a_receipt.json` | Stage A technical smoke | `7aea138f...b742` |
| `artifacts/stage_a1/20260801T091257Z-428e9775/stage_a1_receipt.json` | Stage A.1 ablation | `e23f9917...a2aa` |
| `artifacts/stage_a2/20260801T094412Z-d5f23ea7/stage_a2_receipt.json` | Controlling QI confirmation | `61f63ce2...e38c3` |
| `artifacts/post_a2_diagnostics/diagnostic-fe0f5bbf-61f63ce2/diagnostic_receipt.json` | Stable-regime diagnostic | `009b78b4...2d9` |
| `artifacts/candidate_ceiling_b0/b0-4d48dc5f-61f63ce2/candidate_ceiling_receipt.json` | Candidate-ceiling decision | `5100160a...e0c` |
| `artifacts/candidate_ceiling_b0/b0-4d48dc5f-61f63ce2/integrity_receipt.json` | Hash, alignment, and firewall proof | verified |

The Stage A receipts and Stage A.2 train/validation candidate caches were
byte-identical before and after post-run diagnostics and Stage B.0.

## 10. Closure statement

As of this memo's effective date, QIR-Route on the frozen CSConDa protocol is a
completed negative result. The implementation is technically verified, but the
QI method is not scientifically promoted. Candidate-ceiling recovery is not
sufficient under the preregistered gate. Stage A.3 and Stage B.1 are closed, and
the test partition remains sealed.

Any future study must begin from a new preregistration and must cite this memo as
the closure record for `QIR-CSCONDA-RDM-001`.
