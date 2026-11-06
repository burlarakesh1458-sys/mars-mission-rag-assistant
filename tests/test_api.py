import pytest
from conftest import FakeDense, FakeLLM, FakeReranker
from fastapi.testclient import TestClient

from marsrag import api
from marsrag.config import Settings
from marsrag.pipeline import RAGPipeline
from marsrag.retrieval import HybridRetriever


@pytest.fixture
def client(chunks, monkeypatch):
    s = Settings()
    s.k_context = 3
    pipe = RAGPipeline(settings=s, retriever=HybridRetriever(chunks, FakeDense(chunks), FakeReranker()),
                       llm=FakeLLM("The ratio is 4:1 [1]."), use_reranker=True, refusal_threshold=1.5)
    monkeypatch.setattr(api, "get_pipeline", lambda: pipe)
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ask_returns_answer_with_citations(client):
    r = client.post("/ask", json={"question": "Sabatier reaction hydrogen carbon dioxide ratio"})
    body = r.json()
    assert r.status_code == 200 and not body["refused"]
    assert body["citations"][0]["url"].startswith("https://ntrs.nasa.gov/citations/")
    assert "p. " in body["citations"][0]["label"]


def test_ask_refuses_off_topic(client):
    body = client.post("/ask", json={"question": "best pizza in Naples?"}).json()
    assert body["refused"] and body["citations"] == []


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_search_endpoint(client):
    r = client.get("/search", params={"q": "storm shelter", "k": 2})
    assert r.status_code == 200 and len(r.json()["hits"]) == 2


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Mars Mission" in r.text


def test_reports_lists_corpus(client):
    r = client.get("/reports")
    assert r.status_code == 200
    assert all(x["url"].startswith("https://ntrs.nasa.gov/citations/") for x in r.json()["reports"])
