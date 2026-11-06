"""Ragas evaluation of the end-to-end answers in reports/answers_<config>.jsonl.

Metrics
  * non-LLM (all answerable questions, free): Ragas NonLLMContextPrecisionWithReference and
    NonLLMContextRecall, comparing retrieved chunks to the gold evidence chunk(s) by string
    similarity.
  * LLM-judged (Groq judge model, rate limited): Ragas Faithfulness, LLMContextPrecisionWithReference,
    LLMContextRecall. Scored per sample and appended to reports/ragas_samples_<config>.jsonl so
    the run resumes after a rate limit. --max-llm-samples bounds the token spend.

    python -m marsrag.eval.ragas_eval --config rerank --judge openai/gpt-oss-120b --max-llm-samples 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os

from ..config import ROOT, get_settings
from ..ingest import load_chunks
from .metrics import gold_sources, load_jsonl

REPORTS = ROOT / "reports"


def _shim_vertexai() -> None:
    """ragas 0.4.3 imports VertexAI classes that langchain-community 0.4 removed. We never use
    Vertex, so register placeholder classes if the real ones are missing."""
    import sys
    import types

    try:
        from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
    except ImportError:
        mod = types.ModuleType("langchain_community.chat_models.vertexai")
        mod.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[mod.__name__] = mod
    import langchain_community.llms as llms

    try:
        llms.VertexAI  # noqa: B018
    except (ImportError, AttributeError):
        llms.VertexAI = type("VertexAI", (), {})


def reference_contexts(row: dict, chunks_by_loc: dict) -> list[str]:
    ctx = []
    for d, p in gold_sources(row):
        ctx.extend(chunks_by_loc.get((d, p), []))
    return ctx


def make_judge(model: str):
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    kwargs = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
    llm = ChatOpenAI(model=model, base_url=get_settings().llm_base_url, api_key=os.environ.get("GROQ_API_KEY", "x"),
                     temperature=0, max_tokens=2048, max_retries=8, timeout=120, **kwargs)
    return LangchainLLMWrapper(llm)


async def score_sample(metrics: dict, sample) -> dict:
    out = {}
    for name, m in metrics.items():
        try:
            v = await m.single_turn_ascore(sample)
            out[name] = None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        except Exception as e:  # noqa: BLE001
            out[name] = None
            out[f"{name}_error"] = str(e)[:200]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rerank")
    ap.add_argument("--judge", default="openai/gpt-oss-120b")
    ap.add_argument("--max-llm-samples", type=int, default=40)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--faithfulness-only", action="store_true",
                    help="LLM-judge only Faithfulness (saves judge quota; retrieval uses the non-LLM metrics)")
    args = ap.parse_args(argv)

    _shim_vertexai()
    from ragas import SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        NonLLMContextPrecisionWithReference,
        NonLLMContextRecall,
    )

    s = get_settings()
    by_loc: dict = {}
    for c in load_chunks(s.chunks_path):
        by_loc.setdefault((c.doc_id, c.page), []).append(c.text)
    rows = [r for r in load_jsonl(REPORTS / f"answers_{args.config}.jsonl") if r["answerable"]]
    samples_path = REPORTS / f"ragas_samples_{args.config}.jsonl"
    done = {r["id"]: r for r in load_jsonl(samples_path)} if samples_path.exists() else {}

    nonllm = {"nonllm_context_precision": NonLLMContextPrecisionWithReference(),
              "nonllm_context_recall": NonLLMContextRecall()}
    llm_metrics = {}
    if not args.no_llm:
        judge = make_judge(args.judge)
        llm_metrics = {"faithfulness": Faithfulness(llm=judge),
                       "llm_context_precision": LLMContextPrecisionWithReference(llm=judge),
                       "llm_context_recall": LLMContextRecall(llm=judge)}
        if args.faithfulness_only:
            llm_metrics = {"faithfulness": llm_metrics["faithfulness"]}

    n_llm = sum(1 for r in done.values() if "faithfulness" in r)
    with samples_path.open("a") as f:
        for r in rows:
            prev = done.get(r["id"], {})
            want_llm = bool(llm_metrics) and "faithfulness" not in prev and n_llm < args.max_llm_samples
            if want_llm and r["refused"] and set(llm_metrics) == {"faithfulness"}:
                want_llm = False  # nothing to judge for a declined answer
            if prev and not want_llm:
                continue
            sample = SingleTurnSample(
                user_input=r["question"], response=r["answer"],
                retrieved_contexts=[c["text"] for c in r["contexts"]],
                reference=r["reference"], reference_contexts=reference_contexts(r, by_loc))
            rec = {"id": r["id"], "refused": r["refused"]}
            rec.update(prev)
            if not prev:
                rec.update(asyncio.run(score_sample(nonllm, sample)))
            if want_llm:
                m = dict(llm_metrics)
                if r["refused"]:  # faithfulness is undefined for a refusal; still judge retrieval
                    m.pop("faithfulness")
                res = asyncio.run(score_sample(m, sample))
                if any(k.endswith("_error") for k in res):
                    print(f"  {r['id']} judge error: {[v for k, v in res.items() if k.endswith('_error')]}")
                    if all(res.get(k) is None for k in m):
                        print("  stopping (judge unavailable / rate limited); re-run to resume")
                        break
                rec.update(res)
                rec["faithfulness"] = res.get("faithfulness")  # marks the sample as LLM-judged
                n_llm += 1
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[r["id"]] = rec
            print(f"  {r['id']} " + " ".join(f"{k}={v:.2f}" for k, v in rec.items()
                                            if isinstance(v, float)))
    summary = summarize(load_jsonl(samples_path), args.judge)
    out = REPORTS / "ragas_eval.json"
    allres = json.loads(out.read_text()) if out.exists() else {}
    allres[args.config] = summary
    out.write_text(json.dumps(allres, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def summarize(samples: list[dict], judge: str) -> dict:
    latest = {}
    for srec in samples:  # later lines supersede earlier ones for the same id
        latest[srec["id"]] = {**latest.get(srec["id"], {}), **srec}
    vals = list(latest.values())
    out = {"judge_model": judge, "n_samples": len(vals)}
    for k in ("nonllm_context_precision", "nonllm_context_recall", "faithfulness",
              "llm_context_precision", "llm_context_recall"):
        xs = [v[k] for v in vals if isinstance(v.get(k), float)]
        out[k] = {"mean": sum(xs) / len(xs) if xs else None, "n": len(xs)}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
