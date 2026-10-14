#!/usr/bin/env python3
"""Re-judge the five saved baseline systems' predictions with
bytedance-seed/seed-2.0-mini via OpenRouter, using the paper's exact
LLM_JUDGE_PROMPT. Produces a unified judge table comparable with the
new IBMR run (same judge, same 346 questions, saved predictions only —
baselines are not re-run)."""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

RESULTS = REPO_ROOT / "results" / "full_benchmark_v2"
OUT_DIR = REPO_ROOT / "results" / "ibmr_run"
SOURCE_FILE = RESULTS / "benchmark_results_complete.json"
OUT_FILE = OUT_DIR / "baselines_rejudged_seed2mini.json"

JUDGE_MODEL = "bytedance-seed/seed-2.0-mini"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

LLM_JUDGE_SYSTEM = "You are an expert grader evaluating agricultural domain answers."
LLM_JUDGE_PROMPT = """Evaluate if the generated answer is correct compared to the gold answer.

Question: {question}
Gold Answer: {gold_answer}
Generated Answer: {response}

Guidelines:
- Be generous: if the answer captures the key information, mark as CORRECT
- For disease names, treatments, or technical terms, allow synonyms
- For dates/times, flexible matching (e.g., "last week" vs specific date is OK if contextually correct)
- Partial matches that contain the key information are CORRECT

Return ONLY a JSON object: {{"label": "CORRECT"}} or {{"label": "WRONG"}}"""


async def judge_answer(client, question, gold, response):
    prompt = LLM_JUDGE_PROMPT.format(question=question, gold_answer=gold, response=response)
    for attempt in range(3):
        try:
            r = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": LLM_JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0,
            )
            content = r.choices[0].message.content or ""
            match = re.search(r'\{\s*"label"\s*:\s*"(\w+)"\s*\}', content)
            if match:
                return match.group(1).lower() == "correct"
            return "CORRECT" in content.upper() and "WRONG" not in content.upper()
        except Exception as e:
            if attempt == 2:
                print("\n[judge error] {}".format(e), flush=True)
            await asyncio.sleep(2 * (attempt + 1))
    return False


async def main():
    client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE)
    data = json.loads(SOURCE_FILE.read_text())

    # Flatten: one job per (system, dataset, question)
    jobs = []
    for system_name in ("rag", "nms", "hybrid", "bm25", "memorygraph"):
        for dataset in ("locomo", "hotpotqa"):
            preds = data[system_name][dataset]["predictions"]
            for idx, p in enumerate(preds):
                jobs.append((system_name, dataset, idx, p))

    print("Total judging jobs: {}".format(len(jobs)), flush=True)

    # Checkpointing: resume from partial output
    verdicts = {}
    if OUT_FILE.exists():
        verdicts = json.loads(OUT_FILE.read_text())
        print("Resuming with {} existing verdicts".format(len(verdicts)), flush=True)

    sem = asyncio.Semaphore(8)
    done = [0]
    t0 = time.time()

    async def run_job(job):
        system_name, dataset, idx, p = job
        key = "{}::{}::{}".format(system_name, dataset, idx)
        if key in verdicts:
            return
        async with sem:
            ok = await judge_answer(client, p["question"], p["reference"], p["prediction"])
        verdicts[key] = bool(ok)
        done[0] += 1
        if done[0] % 100 == 0:
            print("[{}/{}] {:.0f}s".format(done[0], len(jobs), time.time() - t0), flush=True)

    async def guarded(job):
        try:
            await run_job(job)
        except Exception as e:
            print("[job error] {}".format(e), flush=True)

    await asyncio.gather(*[guarded(j) for j in jobs])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(verdicts, indent=0))

    # Aggregate unified table
    summary = {}
    for system_name in ("rag", "nms", "hybrid", "bm25", "memorygraph"):
        summary[system_name] = {}
        for dataset in ("locomo", "hotpotqa"):
            preds = data[system_name][dataset]["predictions"]
            keys = ["{}::{}::{}".format(system_name, dataset, i) for i in range(len(preds))]
            oks = [verdicts.get(k) for k in keys if k in verdicts]
            if oks:
                summary[system_name][dataset] = {
                    "n": len(oks),
                    "acc": round(sum(oks) / len(oks), 4),
                }
    OUT_FILE.with_name("baselines_rejudged_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
