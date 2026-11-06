import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marsrag.ingest import Chunk  # noqa: E402
from marsrag.llm import LLMResponse  # noqa: E402


def mk(doc, page, i, text, title=None):
    return Chunk(chunk_id=f"{doc}:p{page}:c{i}", doc_id=doc, title=title or f"Report {doc}",
                 url=f"https://ntrs.nasa.gov/citations/{doc}", page=page, text=text, topic="t")


@pytest.fixture
def chunks():
    return [
        mk("100", 1, 0, "Galactic cosmic rays dominate the radiation dose during Mars transit; shielding with polyethylene helps."),
        mk("100", 2, 0, "Solar particle events can be mitigated by a storm shelter inside the habitat."),
        mk("200", 3, 0, "The Sabatier reaction combines hydrogen and carbon dioxide in a 4:1 ratio to make methane and water."),
        mk("300", 5, 0, "Supersonic retropropulsion is required to land human-scale payloads of 20 tonnes on Mars."),
        mk("400", 7, 0, "Crew members face isolation and communication delays of up to 22 minutes one way."),
    ]


class FakeLLM:
    def __init__(self, text="Answer [1].", fail=False):
        self.text, self.fail, self.calls = text, fail, 0

    def chat(self, messages, max_tokens=None, use_cache=True):
        self.calls += 1
        if self.fail:
            raise RuntimeError("llm down")
        return LLMResponse(text=self.text, model="fake", provider="fake", latency_s=0.01)


class FakeDense:
    """Dense index stand-in: ranks chunks by word overlap with the query."""

    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, k):
        q = set(query.lower().split())
        scored = sorted(self.chunks, key=lambda c: -len(q & set(c.text.lower().split())))
        return [(c.chunk_id, 1.0) for c in scored[:k]]


class FakeReranker:
    def __init__(self, score_fn=None):
        self.score_fn = score_fn or (lambda q, t: float(len(set(q.lower().split()) & set(t.lower().split()))))

    def score(self, query, texts):
        return [self.score_fn(query, t) for t in texts]
