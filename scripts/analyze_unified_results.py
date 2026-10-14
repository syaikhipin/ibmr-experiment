#!/usr/bin/env python3
"""Build the unified IBMR-vs-baselines table with Wilson 95% CIs from
results/ibmr_run/{ibmr_predictions.json, baselines_rejudged_seed2mini.json}.
Also emits per-category breakdown for locomo and iteration/confidence stats."""

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "ibmr_run"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(100 * (centre - half), 1), round(100 * (centre + half), 1))


def fmt(k, n):
    if n == 0:
        return "—"
    lo, hi = wilson(k, n)
    return "{:.1f}% [{}–{}]".format(100 * k / n, lo, hi)


def main():
    ibmr = json.loads((OUT / "ibmr_predictions.json").read_text())
    preds = ibmr["predictions"]
    judged_file = OUT / "ibmr_judged.json"
    if judged_file.exists():
        preds = json.loads(judged_file.read_text())["predictions"]

    baselines = json.loads((OUT / "baselines_rejudged_seed2mini.json").read_text())
    src = json.loads(
        (REPO_ROOT / "results/full_benchmark_v2/benchmark_results_complete.json").read_text()
    )

    systems = {"IBMR": None}
    # IBMR rows
    ibmr_rows = {(p["dataset"], i): p for i, p in enumerate(preds)}

    print("=" * 78)
    print("UNIFIED JUDGE TABLE — bytedance-seed/seed-2.0-mini, paper judge prompt")
    print("=" * 78)

    table = {}
    # IBMR accuracy per dataset
    for ds in ("locomo", "hotpotqa"):
        rows = [p for p in preds if p["dataset"] == ds]
        k = sum(1 for p in rows if p.get("judge_correct"))
        n = len(rows)
        table.setdefault("IBMR", {})[ds] = (k, n)

    for sysname in ("rag", "nms", "hybrid", "bm25", "memorygraph"):
        for ds in ("locomo", "hotpotqa"):
            n_items = len(src[sysname][ds]["predictions"])
            keys = ["{}::{}::{}".format(sysname, ds, i) for i in range(n_items)]
            oks = [baselines[k] for k in keys if k in baselines]
            table.setdefault(sysname.upper(), {})[ds] = (sum(oks), len(oks))

    label = {"locomo": "AgriConvMem", "hotpotqa": "AgriMultiHop"}
    header = "{:<14} {:>26} {:>26}".format("System", label["locomo"] + " (n=146)", label["hotpotqa"] + " (n=200)")
    print(header)
    print("-" * 78)
    for sysname in ("IBMR", "RAG", "NMS", "HYBRID", "BM25", "MEMORYGRAPH"):
        row = table.get(sysname, {})
        cells = []
        for ds in ("locomo", "hotpotqa"):
            k, n = row.get(ds, (0, 0))
            cells.append("{:>26}".format(fmt(k, n)))
        print("{:<14} {}".format(sysname, " ".join(cells)))

    # IBMR logs
    logs = ibmr.get("logs", {})
    iters = logs.get("iterations", [])
    confs = logs.get("confidence", [])
    times = logs.get("search_time", [])
    if iters:
        print("\nIBMR loop statistics (n={}):".format(len(iters)))
        print("  mean iterations:        {:.2f} (K_max=5)".format(sum(iters) / len(iters)))
        print("  budget-exhaustion rate: {:.1f}%".format(100 * sum(1 for i in iters if i >= 5) / len(iters)))
        print("  mean confidence:        {:.3f} (theta=0.9)".format(sum(confs) / len(confs)))
        print("  mean diagnose latency:  {:.1f} ms".format(1000 * sum(times) / len(times)))

    # Per-category (locomo)
    print("\nIBMR by AgriConvMem category:")
    cats = {}
    for p in preds:
        if p["dataset"] != "locomo":
            continue
        c = p["category"]
        cats.setdefault(c, [0, 0])
        cats[c][1] += 1
        if p.get("judge_correct"):
            cats[c][0] += 1
    for c, (k, n) in sorted(cats.items()):
        print("  {:<14} {}  (n={})".format(c, fmt(k, n), n))

    out_path = OUT / "unified_table.json"
    out_path.write_text(json.dumps({
        "table": {s: {ds: {"k": k, "n": n, "acc": round(k / n, 4) if n else None,
                          "wilson95": list(wilson(k, n))}
                      for ds, (k, n) in d.items()}
                  for s, d in table.items()},
        "ibmr_logs": {k: {
            "mean": round(sum(v) / len(v), 4) if v else None,
            "exhausted_5": round(sum(1 for x in iters if x >= 5) / len(iters), 4) if k == "iterations" and iters else None,
        } for k, v in (("iterations", iters), ("confidence", confs), ("search_time", times))},
    }, indent=1))
    print("\nSaved: {}".format(out_path))


if __name__ == "__main__":
    main()
