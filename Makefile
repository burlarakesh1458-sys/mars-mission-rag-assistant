PY ?= python
export PYTHONPATH := src
export OMP_NUM_THREADS := 1

.PHONY: data ingest index golden calibrate eval-retrieval eval-e2e eval-ragas eval-gate test lint serve docker

data:            ## download the NTRS corpus (60 PDFs, ~110 MB) and write data/manifest.jsonl
	$(PY) scripts/fetch_corpus.py --target 60
ingest:          ## PDFs -> page-bounded chunks
	$(PY) -m marsrag.cli ingest
index:           ## chunks -> Chroma (bge-small)
	$(PY) -m marsrag.cli index
golden:          ## rebuild eval/golden.jsonl + eval/calibration.jsonl from reviewed candidates
	$(PY) scripts/finalize_golden.py
calibrate:       ## refusal threshold on the calibration set
	$(PY) scripts/calibrate.py
eval-retrieval:  ## BM25 vs dense vs hybrid vs +rerank (no LLM)
	$(PY) -m marsrag.eval.run_retrieval
eval-e2e:        ## answers + citation / refusal accuracy + latency (resumable)
	$(PY) -m marsrag.eval.run_e2e --config rerank
eval-ragas:      ## Ragas context precision / recall / faithfulness
	$(PY) -m marsrag.eval.ragas_eval --config rerank
eval-gate:       ## offline CI gate: BM25 retrieval + committed results thresholds
	$(PY) -m marsrag.eval.run_retrieval --bm25-only --min-doc-hit5 0.9
	$(PY) scripts/check_eval_gate.py
test:
	$(PY) -m pytest -q
lint:
	ruff check src tests scripts
serve:
	$(PY) -m marsrag.cli serve
docker:
	docker build -t marsrag .
