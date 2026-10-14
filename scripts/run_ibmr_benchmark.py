#!/usr/bin/env python3
"""IBMR (CLS bicameral) adapter + runner for the AgriMemSynth benchmark.

Ingest via cls-paper EpisodicKnowledgeGraph.fast_write; query via the
iterative bidirectional loop (CLSMemorySystem.diagnose, K_max=5,
theta=0.9, weighted_vote reconciliation). Final answer generated with the
SAME generator as the paper (deepseek/deepseek-chat-v3-0324 via OpenRouter)
and the SAME ANSWER_PROMPT. Judged with bytedance-seed/seed-2.0-mini.
"""

import asyncio
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_IBMR = REPO_ROOT / "src" / "ibmr"
if str(SRC_IBMR) not in sys.path:
    sys.path.insert(0, str(SRC_IBMR))

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv(REPO_ROOT / ".env")

_spec = importlib.util.spec_from_file_location(
    "scirep_bench", REPO_ROOT / "scripts" / "comprehensive_benchmark_full.py"
)
scirep_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scirep_bench)

calculate_f1 = scirep_bench.calculate_f1
calculate_exact_match = scirep_bench.calculate_exact_match
load_agri_locomo = scirep_bench.load_agri_locomo
load_agri_hotpotqa = scirep_bench.load_agri_hotpotqa
ANSWER_PROMPT = scirep_bench.ANSWER_PROMPT

GENERATOR_MODEL = "deepseek/deepseek-chat-v3-0324"
JUDGE_MODEL = "bytedance-seed/seed-2.0-mini"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

from config import AgentControllerConfig, CLSMemorySystemConfig
from data_model import DiagnosticContext
from model import CLSMemorySystem


def make_cls_system() -> CLSMemorySystem:
    ctrl = AgentControllerConfig(
        max_iterative_cycles=5,
        early_exit_confidence=0.9,
        reconciliation_method="weighted_vote",
        enable_semantic_prior_routing=True,
        enable_episodic_revision=True,
    )
    return CLSMemorySystem(CLSMemorySystemConfig(agent_controller=ctrl))


class IBMRBenchmarkAdapter:
    def __init__(self):
        self.name = "IBMR"
        self.system = make_cls_system()
        self.system.eval()
        self.field_id = "agri"
        self.gen = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE)
        self.logs = {"iterations": [], "confidence": [], "search_time": []}

    def reset(self, sample_id: str = None):
        self.system.reset()
        self.field_id = "agri_{}".format(sample_id or "anon")

    def _ts(self, base, offset):
        try:
            dt = datetime.fromisoformat(str(base).replace("Z", "+00:00"))
        except Exception:
            dt = datetime(2024, 1, 1, 8, 0, 0)
        return dt + timedelta(minutes=2 * offset)

    def add_conversation(self, conversation, metadata=None):
        ts_base = (metadata or {}).get("timestamp") or "2024-01-01T08:00:00"
        for i, turn in enumerate(conversation):
            text = "{}: {}".format(turn.get("speaker", "farmer"), turn.get("text", ""))
            self.system.episodic_kg.fast_write({
                "type": "observation",
                "timestamp": self._ts(ts_base, i),
                "field_id": self.field_id,
                "query": text,
                "summary": text,
            })
        self._embeddings_stale = True

    def _prewarm_semantic_index(self):
        """Mirror ConsolidationScheduler.step's embedding refresh: encode every
        pending subgraph with the GCN and cache for semantic-prior retrieval.
        (The scheduler's maturity gate — repeat_count>=2, 60-min age — never
        fires within a benchmark session, so refresh directly.)"""
        kg = self.system.episodic_kg
        sem = self.system.semantic_memory
        embeds = [sem.infer_pattern(sg).pattern_embed for sg in kg.pending_consolidation]
        if embeds:
            kg.update_subgraph_embeddings(embeds)

    def search(self, query, top_k=10):
        t0 = time.time()
        if getattr(self, "_embeddings_stale", True):
            self._prewarm_semantic_index()
            self._embeddings_stale = False
        ctx = DiagnosticContext(field_id=self.field_id)
        resp = self.system.diagnose(query, ctx)
        elapsed = time.time() - t0
        # Skip the self-observation diagnose() writes for the current query
        skip_prefixes = ("Query: ",)
        parts = [
            p.get("claim", "") for p in (resp.provenance or [])
            if p.get("source") == "episodic_kg" and not p.get("claim", "").startswith(skip_prefixes)
        ]
        parts += [
            e.get("content", "") for e in (resp.evidence or [])
            if e.get("type") == "semantic" and e.get("content")
        ]
        context = "\n".join([p for p in parts if p]) or "(no retrieved context)"
        self.logs["iterations"].append(resp.num_iterations)
        self.logs["confidence"].append(round(resp.confidence, 4))
        self.logs["search_time"].append(round(elapsed, 4))
        return context, elapsed

    def generate_response(self, question, context):
        prompt = ANSWER_PROMPT.replace("{{CONTEXT}}", context).replace("{{QUESTION}}", question)
        try:
            r = self.gen.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100,
            )
            return (r.choices[0].message.content or "").strip() or "unknown"
        except Exception as e:
            print("\n[gen error] {}".format(e))
            return "error"

    def query(self, question, top_k=10):
        context, _ = self.search(question, top_k)
        answer = self.generate_response(question, context)
        return answer, context


JUDGE_SYSTEM = "You are an expert grader evaluating agricultural domain answers."
# Exact prompt from comprehensive_benchmark_full.py (LLM_JUDGE_PROMPT)
JUDGE_PROMPT = """Evaluate if the generated answer is correct compared to the gold answer.

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
    prompt = JUDGE_PROMPT.format(question=question, gold_answer=gold, response=response)
    for attempt in range(3):
        try:
            r = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
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
                print("\n[judge error] {}".format(e))
            await asyncio.sleep(2 * (attempt + 1))
    return False


def load_questions(dataset):
    """Yield task dicts: ingest turns + question + reference + meta."""
    tasks = []
    if dataset in ("locomo", "both"):
        for s in load_agri_locomo(str(REPO_ROOT / "data/text/agri_locomo_v2/test.json")):
            turns = []
            conv = s.get("conversation", {})
            for key, value in conv.items():
                if key.startswith("session_") and isinstance(value, list):
                    ts = conv.get(key + "_date_time", "")
                    for t in value:
                        turns.append((t, ts))
            for qa in s.get("qa", []):
                tasks.append({
                    "dataset": "locomo",
                    "sample_id": s.get("sample_id", ""),
                    "turns": turns,
                    "question": qa["question"],
                    "reference": str(qa["answer"]),
                    "category": qa.get("category_name", str(qa.get("category", 0))),
                })
    if dataset in ("hotpotqa", "both"):
        for s in load_agri_hotpotqa(str(REPO_ROOT / "data/text/agri_hotpotqa_v4/test.json")):
            turns = []
            for para in s.get("context", []):
                title, text = para.get("title", ""), para.get("text", "")
                turns.append(({"speaker": "document", "text": "{}: {}".format(title, text) if title else text}, ""))
            tasks.append({
                "dataset": "hotpotqa",
                "sample_id": s.get("id", ""),
                "turns": turns,
                "question": s.get("question", ""),
                "reference": str(s.get("answer", "")),
                "category": s.get("type", "bridge"),
            })
    return tasks


async def main(dataset="both", limit=None):
    client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE)
    ibmr = IBMRBenchmarkAdapter()
    tasks = load_questions(dataset)
    if limit:
        tasks = tasks[:limit]
    print("Total questions: {}".format(len(tasks)))

    out_dir = REPO_ROOT / "results" / "ibmr_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: reload prior predictions + logs
    pred_file = out_dir / "ibmr_predictions.json"
    logs = {"iterations": [], "confidence": [], "search_time": []}
    predictions = []
    done_keys = set()
    if pred_file.exists():
        prev = json.loads(pred_file.read_text())
        predictions = prev.get("predictions", [])
        logs = prev.get("logs", logs)
        done_keys = {"{}::{}".format(p["sample_id"], p["question"]) for p in predictions}
        print("Resuming: {} predictions already done".format(len(predictions)))

    # Per-question watchdog: diagnose() can rarely wedge in a pure-Python
    # spin (observed once at Q130); abort -> mark failed -> continue.
    WATCHDOG_S = 90
    import signal

    def _alarm(signum, frame):
        raise TimeoutError("diagnose watchdog ({}s)".format(WATCHDOG_S))

    t_start = time.time()
    current_sample = None
    n_fail = 0
    for i, task in enumerate(tasks):
        key = "{}::{}".format(task["sample_id"], task["question"])
        if key in done_keys:
            continue
        if task["sample_id"] != current_sample:
            # Ingest each sample once; answer all its QAs against it
            current_sample = task["sample_id"]
            ibmr.reset(current_sample)
            by_ts = {}
            for t, ts in task["turns"]:
                by_ts.setdefault(ts, []).append(t)
            for ts, group in by_ts.items():
                ibmr.add_conversation(group, {"timestamp": ts})
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(WATCHDOG_S)
        try:
            answer, _ = ibmr.query(task["question"])
            ibmr.logs = logs  # search() appends into logs
            predictions.append({
                "sample_id": task["sample_id"],
                "dataset": task["dataset"],
                "category": task["category"],
                "question": task["question"],
                "reference": task["reference"],
                "prediction": answer,
            })
            done_keys.add(key)
        except TimeoutError as e:
            n_fail += 1
            print("\n[watchdog] Q{} {}: {}".format(i, e, task["question"][:60]), flush=True)
            predictions.append({
                "sample_id": task["sample_id"],
                "dataset": task["dataset"],
                "category": task["category"],
                "question": task["question"],
                "reference": task["reference"],
                "prediction": "error:watchdog",
            })
            done_keys.add(key)
            ibmr.reset(current_sample)  # fresh system after a wedged state
            current_sample = None       # force re-ingest on next question
        finally:
            signal.alarm(0)

        # Checkpoint every 10 questions
        if (i + 1) % 10 == 0 or i + 1 == len(tasks):
            pred_file.write_text(json.dumps({
                "generator": GENERATOR_MODEL, "judge": JUDGE_MODEL,
                "logs": logs, "predictions": predictions}, indent=1))
            print("[{}/{}] {:.0f}s elapsed | fails={} | last-10 iters {}".format(
                i + 1, len(tasks), time.time() - t_start, n_fail,
                logs["iterations"][-10:]), flush=True)

    pred_file.write_text(json.dumps({
        "generator": GENERATOR_MODEL, "judge": JUDGE_MODEL,
        "logs": logs, "predictions": predictions}, indent=1))
    print("Predictions saved: {}".format(pred_file))

    # Judge phase (concurrent, semaphore=8; resume-aware)
    judged_file = out_dir / "ibmr_judged.json"
    judged = {}
    if judged_file.exists():
        for p in json.loads(judged_file.read_text())["predictions"]:
            judged["{}::{}".format(p["sample_id"], p["question"])] = p.get("judge_correct")

    sem = asyncio.Semaphore(8)

    async def judge_one(p):
        key = "{}::{}".format(p["sample_id"], p["question"])
        if key in judged:
            p["judge_correct"] = judged[key]
            return
        async with sem:
            ok = await judge_answer(client, p["question"], p["reference"], p["prediction"])
        p["judge_correct"] = bool(ok)
        judged[key] = p["judge_correct"]

    await asyncio.gather(*[judge_one(p) for p in predictions])

    # Aggregate (watchdog failures judged as incorrect; reported separately)
    def agg(rows):
        n = len(rows)
        if not n:
            return {}
        return {
            "n": n,
            "judge_acc": round(sum(1 for r in rows if r.get("judge_correct")) / n, 4),
            "f1": round(sum(calculate_f1(r["prediction"], r["reference"]) for r in rows) / n, 4),
            "em": round(sum(calculate_exact_match(r["prediction"], r["reference"]) for r in rows) / n, 4),
            "mean_iters": round(sum(logs["iterations"]) / len(logs["iterations"]), 3)
            if logs["iterations"] else None,
        }

    locomo_rows = [p for p in predictions if p["dataset"] == "locomo"]
    hotpot_rows = [p for p in predictions if p["dataset"] == "hotpotqa"]
    n_err = sum(1 for p in predictions if p["prediction"] == "error:watchdog")
    summary = {
        "generator": GENERATOR_MODEL,
        "judge": JUDGE_MODEL,
        "config": {"K_max": 5, "theta": 0.9, "reconciliation": "weighted_vote"},
        "n_watchdog_failures": n_err,
        "locomo_overall": agg(locomo_rows),
        "hotpotqa_overall": agg(hotpot_rows),
        "locomo_by_category": {
            c: agg([r for r in locomo_rows if r["category"] == c])
            for c in sorted({r["category"] for r in locomo_rows})
        },
    }
    (out_dir / "ibmr_summary.json").write_text(json.dumps(summary, indent=1))
    judged_file.write_text(json.dumps({"predictions": predictions}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "both"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    asyncio.run(main(dataset, limit))
