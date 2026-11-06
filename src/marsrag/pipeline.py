"""Question -> retrieve -> (refusal gate) -> grounded answer with exact citations."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from .config import Settings, get_settings
from .ingest import load_chunks
from .retrieval import DenseIndex, Embedder, Hit, HybridRetriever, Reranker

REFUSAL_TOKEN = "INSUFFICIENT_CONTEXT"
REFUSAL_MESSAGE = ("I can't answer that from the NASA technical reports in this corpus. "
                   "They don't contain information on this question.")

SYSTEM_PROMPT = f"""You are a research assistant answering questions about human missions to Mars \
using ONLY the numbered NASA technical report excerpts provided.
Rules:
1. Use only facts stated in the excerpts. Do not use outside knowledge, even if you know the answer.
2. After every sentence that states a fact, cite the excerpt(s) it came from like [1] or [2][3].
3. If the excerpts do not contain the information needed to answer, reply with exactly {REFUSAL_TOKEN} and nothing else.
4. Be concise: 2-5 sentences. Keep numbers and units exactly as written in the excerpts."""

_CITE = re.compile(r"\[(\d{1,2})\]")


@dataclass
class Citation:
    n: int
    doc_id: str
    title: str
    page: int
    url: str
    chunk_id: str

    @property
    def label(self) -> str:
        return f"{self.title}, p. {self.page} ({self.url})"


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[Citation]
    contexts: list[dict]
    timings_ms: dict = field(default_factory=dict)
    llm: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citations"] = [asdict(c) | {"label": c.label} for c in self.citations]
        return d


def build_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        blocks.append(f"[{i}] {c.title} (NTRS {c.doc_id}, page {c.page})\n{c.text}")
    return "\n\n".join(blocks)


def parse_citations(text: str, hits: list[Hit]) -> tuple[list[Citation], list[int]]:
    """Return (valid citations in first-use order, invalid citation numbers)."""
    seen, valid, invalid = set(), [], []
    for m in _CITE.finditer(text):
        n = int(m.group(1))
        if n in seen:
            continue
        seen.add(n)
        if 1 <= n <= len(hits):
            c = hits[n - 1].chunk
            valid.append(Citation(n, c.doc_id, c.title, c.page, c.url, c.chunk_id))
        else:
            invalid.append(n)
    return valid, invalid


def strip_invalid(text: str, invalid: list[int]) -> str:
    for n in invalid:
        text = text.replace(f"[{n}]", "")
    return text


def is_refusal(text: str) -> bool:
    t = text.strip()
    return REFUSAL_TOKEN in t or not t


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, retriever: HybridRetriever | None = None,
                 llm=None, use_reranker: bool | None = None, refusal_threshold: float | None = None,
                 rerank_mode: str | None = None):
        self.s = settings or get_settings()
        self.use_reranker = self.s.use_reranker if use_reranker is None else use_reranker
        self.refusal_threshold = self.s.refusal_threshold if refusal_threshold is None else refusal_threshold
        self.rerank_mode = rerank_mode or self.s.rerank_mode
        if retriever is None:
            chunks = load_chunks(self.s.chunks_path)
            dense = DenseIndex(self.s.index_dir, Embedder(self.s.embed_model))
            reranker = Reranker(self.s.rerank_model) if self.use_reranker else None
            retriever = HybridRetriever(chunks, dense, reranker, self.s.k_dense, self.s.k_bm25,
                                        self.s.rrf_k, self.s.k_rerank_pool)
        self.retriever = retriever
        if llm is None:
            from .llm import default_client
            llm = default_client(self.s)
        self.llm = llm

    def retrieve(self, question: str) -> tuple[list[Hit], float]:
        t0 = time.perf_counter()
        hits = self.retriever.retrieve(question, k=self.s.k_context, mode="hybrid", rerank=self.use_reranker,
                                       rerank_mode=self.rerank_mode)
        return hits, (time.perf_counter() - t0) * 1000

    def gate(self, hits: list[Hit]) -> str | None:
        if not hits:
            return "no_hits"
        if self.use_reranker and hits[0].rerank_score is not None and hits[0].rerank_score < self.refusal_threshold:
            return f"low_relevance(top_rerank={hits[0].rerank_score:.2f}<{self.refusal_threshold:.2f})"
        return None

    def answer(self, question: str) -> Answer:
        t_all = time.perf_counter()
        hits, t_ret = self.retrieve(question)
        contexts = [{"chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id, "title": h.chunk.title,
                     "page": h.chunk.page, "url": h.chunk.url, "text": h.chunk.text,
                     "rerank_score": h.rerank_score} for h in hits]
        reason = self.gate(hits)
        if reason:
            return Answer(question, REFUSAL_MESSAGE, True, reason, [], contexts,
                          {"retrieve": t_ret, "llm": 0.0, "total": (time.perf_counter() - t_all) * 1000})
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Excerpts:\n\n{build_context(hits)}\n\nQuestion: {question}"}]
        t1 = time.perf_counter()
        resp = self.llm.chat(messages)
        t_llm = (time.perf_counter() - t1) * 1000
        llm_meta = {"model": resp.model, "provider": resp.provider, "cached": resp.cached,
                    "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
                    "provider_latency_ms": resp.latency_s * 1000}
        timings = {"retrieve": t_ret, "llm": t_llm, "total": (time.perf_counter() - t_all) * 1000}
        if is_refusal(resp.text):
            return Answer(question, REFUSAL_MESSAGE, True, "llm_declined", [], contexts, timings, llm_meta)
        cits, invalid = parse_citations(resp.text, hits)
        text = strip_invalid(resp.text, invalid)
        return Answer(question, text, False, None, cits, contexts, timings, llm_meta)
