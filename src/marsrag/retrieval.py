"""Hybrid retrieval: BM25 (lexical) + bge-small dense vectors in Chroma, fused with RRF,
then optionally reranked by a MiniLM cross-encoder.

Heavy models are imported lazily so the BM25 path (used by the CI eval gate) runs
without torch.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .ingest import Chunk

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have how i if in into is it "
    "its may might more most much no not of on or our over should so such than that the their them then "
    "there these they this those through to under up was we were what when where which while who why will "
    "with would you your".split()
)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_r 1 / (k + rank_r(d)), ranks 1-based."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass
class Hit:
    chunk: Chunk
    score: float
    sources: dict = field(default_factory=dict)  # per-retriever rank / score for debugging
    rerank_score: float | None = None


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        # title is prepended so report-level terms ("radiation", "EDL") help short chunks
        self._bm25 = BM25Okapi([tokenize(c.title + " " + c.text) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        import numpy as np

        scores = self._bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i].chunk_id, float(scores[i])) for i in top if scores[i] > 0]


class Embedder:
    _cache: dict[str, object] = {}

    def __init__(self, model_name: str):
        self.model_name = model_name

    @property
    def model(self):
        if self.model_name not in Embedder._cache:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
            Embedder._cache[self.model_name] = SentenceTransformer(self.model_name, device="cpu")
        return Embedder._cache[self.model_name]

    def encode_docs(self, texts: list[str], batch_size: int = 32):
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=False)

    def encode_query(self, q: str):
        prefix = BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return self.model.encode([prefix + q], normalize_embeddings=True, show_progress_bar=False)[0]


class DenseIndex:
    """Chroma persistent collection with cosine distance; embeddings computed by us."""

    COLLECTION = "nasa_mars"

    def __init__(self, index_dir: Path, embedder: Embedder):
        import chromadb

        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(index_dir))
        self.col = self.client.get_or_create_collection(self.COLLECTION, metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self.col.count()

    def build(self, chunks: list[Chunk], batch: int = 64, log=print) -> None:
        try:
            self.client.delete_collection(self.COLLECTION)
        except Exception:  # noqa: BLE001
            pass
        self.col = self.client.get_or_create_collection(self.COLLECTION, metadata={"hnsw:space": "cosine"})
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            embs = self.embedder.encode_docs([f"{c.title}\n{c.text}" for c in part])
            self.col.add(
                ids=[c.chunk_id for c in part],
                embeddings=[e.tolist() for e in embs],
                metadatas=[{"doc_id": c.doc_id, "page": c.page} for c in part],
            )
            if log and (i // batch) % 10 == 0:
                log(f"  embedded {i + len(part)}/{len(chunks)}")

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = self.embedder.encode_query(query)
        res = self.col.query(query_embeddings=[q.tolist()], n_results=k)
        return [(cid, 1.0 - float(d)) for cid, d in zip(res["ids"][0], res["distances"][0])]


class Reranker:
    _cache: dict[str, object] = {}

    def __init__(self, model_name: str):
        self.model_name = model_name

    @property
    def model(self):
        if self.model_name not in Reranker._cache:
            import torch
            from sentence_transformers import CrossEncoder

            torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
            Reranker._cache[self.model_name] = CrossEncoder(self.model_name, device="cpu", max_length=512)
        return Reranker._cache[self.model_name]

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts], batch_size=16,
                                                     show_progress_bar=False)]


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], dense: DenseIndex | None, reranker: Reranker | None,
                 k_dense: int = 30, k_bm25: int = 30, rrf_k: int = 60, k_rerank_pool: int = 20):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.bm25 = BM25Index(chunks)
        self.dense = dense
        self.reranker = reranker
        self.k_dense, self.k_bm25, self.rrf_k, self.k_rerank_pool = k_dense, k_bm25, rrf_k, k_rerank_pool

    def retrieve(self, query: str, k: int = 5, mode: str = "hybrid", rerank: bool | None = None) -> list[Hit]:
        """mode in {bm25, dense, hybrid}. rerank=None -> rerank iff a reranker is configured."""
        bm = self.bm25.search(query, self.k_bm25) if mode in ("bm25", "hybrid") else []
        de = self.dense.search(query, self.k_dense) if mode in ("dense", "hybrid") and self.dense else []
        if mode == "hybrid":
            fused = rrf_fuse([[c for c, _ in bm], [c for c, _ in de]], k=self.rrf_k)
        elif mode == "bm25":
            fused = bm
        elif mode == "dense":
            fused = de
        else:
            raise ValueError(f"unknown mode {mode}")
        bm_rank = {c: r for r, (c, _) in enumerate(bm, 1)}
        de_rank = {c: r for r, (c, _) in enumerate(de, 1)}
        hits = [Hit(self.by_id[c], s, {"bm25_rank": bm_rank.get(c), "dense_rank": de_rank.get(c)})
                for c, s in fused if c in self.by_id]
        do_rerank = (self.reranker is not None) if rerank is None else rerank
        if do_rerank and self.reranker is not None and hits:
            pool = hits[: self.k_rerank_pool]
            scores = self.reranker.score(query, [h.chunk.text for h in pool])
            for h, s in zip(pool, scores):
                h.rerank_score = s
            hits = sorted(pool, key=lambda h: -h.rerank_score)
        return hits[:k]
