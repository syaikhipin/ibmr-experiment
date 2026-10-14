#!/usr/bin/env python3
"""Compute per-category AgriConvMem accuracy for IBMR and the baseline
band (min-max across the five baselines) under the unified judge.

Fixes the RESULTS.md v1 defect: the by_category fields in
benchmark_results_complete.json are token-level F1 scores, NOT judge
scores. Baseline category accuracies must be computed by joining the
per-question Seed-2.0-mini verdicts in baselines_rejudged_seed2mini.json
with the category labels in the benchmark predictions.

Outputs: results/ibmr_run/category_bands.json
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "ibmr_run"
SYSTEMS = ["rag", "nms", "hybrid", "bm25", "memorygraph"]


def main():
    src = json.loads(
        (REPO_ROOT / "results/full_benchmark_v2/benchmark_results_complete.json").read_text()
    )
    rej = json.loads((OUT / "baselines_rejudged_seed2mini.json").read_text())
    ibmr = json.loads((OUT / "ibmr_judged.json").read_text())["predictions"]

    # category label per question index (order verified identical across systems)
    cats = [p["category_name"] for p in src["rag"]["locomo"]["predictions"]]

    band = {}
    for s in SYSTEMS:
        preds = src[s]["locomo"]["predictions"]
        for i, cat in enumerate(cats):
            ok = 1 if rej["{}::locomo::{}".format(s, i)] else 0
            band.setdefault(cat, {}).setdefault(s, []).append(ok)

    ibmr_by_cat = {}
    for p in ibmr:
        if p["dataset"] != "locomo":
            continue
        ibmr_by_cat.setdefault(p["category"], []).append(1 if p.get("judge_correct") else 0)

    out = {"judge": "bytedance-seed/seed-2.0-mini", "n": 146, "categories": {}}
    for cat in ["disease_identification", "severity", "temporal", "treatment"]:
        accs = {s: round(100 * sum(v) / len(v), 1) for s, v in band[cat].items()}
        ib = ibmr_by_cat.get(cat, [])
        ib_acc = round(100 * sum(ib) / len(ib), 1) if ib else None
        out["categories"][cat] = {
            "ibmr_acc": ib_acc,
            "ibmr_k": sum(ib),
            "ibmr_n": len(ib),
            "baseline_acc": accs,
            "baseline_min": min(accs.values()),
            "baseline_max": max(accs.values()),
        }

    dest = OUT / "category_bands.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("\nSaved: {}".format(dest))


if __name__ == "__main__":
    main()
