# IBMR Experiment

Controlled evaluation of the **Iterative Bidirectional Memory Router (IBMR)**
against five retrieval and memory baselines on 346 held-out AgriMemSynth
questions (146 AgriConvMem conversational + 200 AgriMultiHop multi-hop).

Paper: `paper/ibmr_eval.tex` (IEEEtran, camera-ready source + compiled PDF).

## Headline results (unified judge: Seed-2.0-mini, generator: DeepSeek-V3)

| System | AgriConvMem (n=146) | AgriMultiHop (n=200) |
|---|---|---|
| IBMR (iterative) | 46.6% [38.7, 54.7] | 77.5% [71.2, 82.7] |
| RAG | 74.0% | 76.5% |
| Hybrid (NMS+RAG) | 71.9% | 74.5% |
| NMS | 68.5% | 78.5% |
| MemoryGraph | 67.8% | 78.5% |
| BM25 + Re-rank | 66.4% | 76.5% |

IBMR sits within the baseline band on multi-hop questions but loses to
single-pass RAG on conversational recall (paired McNemar tests, p < 1e-4
against every baseline). Every completed query exhausted the K_max=5
iteration budget; the weighted-vote confidence plateaued at 0.750 against a
0.90 early-exit threshold.

## Repository layout

```
src/ibmr/            IBMR controller source (episodic KG, semantic GCN,
                     reconciliation loop, runner entry points)
scripts/             run_ibmr_benchmark.py (IBMR run),
                     comprehensive_benchmark_full.py (baseline benchmark),
                     rejudge_baselines.py (unified-judge re-scoring),
                     analyze_unified_results.py (unified table + Wilson CIs),
                     category_bands.py (per-category bands)
configs/             experiment_config.yaml
data/text/           AgriMemSynth text splits used in this run
                     (agri_locomo_v2, agri_hotpotqa_v4)
results/ibmr_run/    IBMR per-question records, judge verdicts, loop
                     telemetry, unified table, run logs
results/full_benchmark_v2/
                     baseline benchmark predictions + category/type tables;
                     baseline_verdicts_seed2mini.json holds the
                     per-question Seed-2.0-mini verdicts
paper/               manuscript source (ibmr_eval.tex, ibmr_eval.bib),
                     figure generator (figures.py), figures
```

## Reproduction

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the environment template and fill in API credentials:

```bash
cp .env.example .env
```

4. Run the baselines (see `source` script header for flags), then the IBMR run:

```bash
python scripts/comprehensive_benchmark_full.py \
  --dataset both --data-dir data/text --split test \
  --techniques nms,rag,hybrid,bm25,memorygraph \
  --output results/full_benchmark_v2/reproduced_benchmark.json
python scripts/run_ibmr_benchmark.py
```

5. Re-score and rebuild the tables:

```bash
python scripts/rejudge_baselines.py
python scripts/analyze_unified_results.py
python scripts/category_bands.py
```

## Notes

- Rerunning the LLM calls requires API credentials and budget; the saved
  per-question records in `results/` reproduce every table in the paper
  without new API spend.
- One of 346 queries was aborted by the 90 s watchdog (nondeterministic
  compute spin) and scored as incorrect; details are in the paper.
- `RESULTS.md` in `results/` documents the run configuration and the
  v1 -> v2 stability note.
