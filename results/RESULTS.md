# IBMR Unified-Judge Comparison: Results (v2 rerun, 2026-09-22)

Source run: `sourcecode/scirep/results/ibmr_run/` (fresh rerun of the
original run; the v1 originals are preserved verbatim in
`results/ibmr_run_v1_backup_20260922/`).
Summary files: `unified_table.json`, `ibmr_summary.json`,
`category_bands.json`, `baselines_rejudged_summary.json`. Full run
trace: `full_run.log`. Per-question records: `ibmr_predictions.json`
(IBMR outputs), `ibmr_judged.json` (judge verdicts),
`baselines_rejudged_seed2mini.json` (baseline predictions re-scored
under the unified judge).

## Run configuration

| Item | Value |
|---|---|
| Generator (all six systems) | deepseek/deepseek-chat-v3-0324 |
| Unified judge (all six systems) | bytedance-seed/seed-2.0-mini |
| Iteration budget K_max | 5 |
| Early-exit threshold theta | 0.9 |
| Reconciliation | weighted_vote |
| Questions | 346 (146 AgriConvMem "locomo" + 200 AgriMultiHop "hotpotqa") |
| Watchdog failures | 1 (sample agri-0154, severity category, 90 s timeout, scored as incorrect) |
| Baselines | rag, nms, hybrid (nms+rag), bm25+rerank, memorygraph |

## Headline accuracies (unified judge)

From `unified_table.json` (k correct / n, acc, Wilson 95% CI in points):

| System | AgriConvMem (n=146) | AgriMultiHop (n=200) |
|---|---|---|
| IBMR (iterative) | 68/146 = 46.6% [38.7, 54.7] | 155/200 = 77.5% [71.2, 82.7] |
| RAG | 108/146 = 74.0% [66.3, 80.4] | 153/200 = 76.5% [70.2, 81.8] |
| Hybrid (NMS+RAG) | 105/146 = 71.9% [64.1, 78.6] | 149/200 = 74.5% [68.0, 80.0] |
| NMS | 100/146 = 68.5% [60.6, 75.5] | 157/200 = 78.5% [72.3, 83.6] |
| MemoryGraph | 99/146 = 67.8% [59.9, 74.8] | 157/200 = 78.5% [72.3, 83.6] |
| BM25 + Re-rank | 97/146 = 66.4% [58.4, 73.6] | 153/200 = 76.5% [70.2, 81.8] |

Readings:

- Conversational: IBMR's interval [38.7, 54.7] is disjoint from every
  baseline's; single-pass RAG leads at 74.0%.
- Multi-hop: IBMR (77.5%) sits within the baseline band — NMS and
  MemoryGraph nominally lead at 78.5%, but intervals overlap broadly;
  the dataset does not separate the top systems (v1's three-way tie at
  78.5% was within this noise).

## IBMR per-category breakdown (AgriConvMem)

From `ibmr_summary.json` `locomo_by_category` (judge_acc = unified-judge
accuracy; f1 and em are token-level metrics from the same predictions):

| Category | n | judge_acc | em | f1 | mean_iters |
|---|---|---|---|---|---|
| Disease identification | 23 | 95.65% | 17.4% | 0.5874 | 5.0 |
| Severity | 41 | 53.66% | 0.0% | 0.1609 | 5.0 |
| Temporal | 47 | 23.40% | 0.0% | 0.1201 | 5.0 |
| Treatment | 35 | 37.14% | 2.9% | 0.1386 | 5.0 |

Baseline per-category bands under the unified judge (from
`category_bands.json`; computed by joining per-question Seed-2.0-mini
verdicts with category labels — **not** from the `by_category` fields
of `benchmark_results_complete.json`, which are token-level F1 scores):

| Category | IBMR | Baseline band (min–max) | Best baseline |
|---|---|---|---|
| Disease identification | 95.7% | 95.7–100.0 | rag, bm25, memorygraph (100%) |
| Severity | 53.7% | 82.9–100.0 | rag (100%) |
| Temporal | 23.4% | 38.3–44.7 | nms, hybrid (44.7%) |
| Treatment | 37.1% | 62.9–74.3 | rag (74.3%) |

IBMR ties the band floor on disease identification (95.7%) and is at
or below every baseline on severity, temporal, and treatment. Under
the unified judge, IBMR leads on **no** category.

## Overall token-level metrics (IBMR)

From `ibmr_summary.json`:

| Dataset | n | judge_acc | f1 | em | mean_iters |
|---|---|---|---|---|---|
| AgriConvMem (locomo) | 146 | 46.58% | 0.2096 | 0.0342 | 5.0 |
| AgriMultiHop (hotpotqa) | 200 | 77.50% | 0.3956 | 0.1350 | 5.0 |

## Loop dynamics

From `unified_table.json` `ibmr_logs` and `full_run.log`:

- Mean iterations: 5.0; every completed query exhausted the K_max=5
  budget (`exhausted_5` = 1.0). The theta=0.9 early exit never fired.
- Mean weighted-vote confidence at termination: 0.7503, i.e. the loop
  plateaued at ~0.750, far below theta=0.9. Termination is uncalibrated
  for this domain.
- Mean search_time per query: 0.0794 s (~79 ms, CPU; telemetry covers
  344 of 346 queries — the watchdog-aborted query and one resume-boundary
  query do not append telemetry).
- Watchdog: 1 failure (sample agri-0154, a severity question, 90 s
  diagnose watchdog), scored as incorrect. Note: v1's watchdog fired on
  a different severity question (Q132), confirming the spin is
  nondeterministic.

## Defects patched before the run

Five defects in the released conversational ingest path were found and
fixed before the original comparison (documented in the release notes):
spatial-index population for dictionary-form events, the query-centre
coordinate fallback, the retrieval type filter excluding observation
nodes, missing subgraph summaries, and tensorless query-subgraph
construction. These are harness findings of the same character as the
five unit-level bugs found in the AGRI-CLS validation study (38 unit
tests, 9 benchmarks, 8 ablations).

## v1 → v2 stability note

The rerun reproduces v1's qualitative picture with small numeric drift,
as expected under a nondeterministic generator+judge stack:
conversational 51.4% → 46.6%, multi-hop 78.5% → 77.5%, category pattern
(strong disease ID, collapsed temporal/treatment) unchanged, loop
telemetry identical (5.0 iters, 0.750 plateau, 1 watchdog failure).
This run-pair is itself evidence for the paper's single-run
nondeterminism caveat.
