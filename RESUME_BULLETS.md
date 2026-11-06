# Resume bullets: Mars Mission Research Assistant

Every number below is measured and comes from `reports/retrieval_eval.json`, `reports/e2e_eval.json`
or `reports/ragas_eval.json` (rendered in `reports/RESULTS.md`). Nothing is projected or rounded up.

* Built a citation-grounded RAG assistant over **60 NASA NTRS reports (1,083 pages, 2,391 page-bounded chunks)**: BM25 + bge-small dense retrieval fused with Reciprocal Rank Fusion, a MiniLM cross-encoder reranker, and an LLM that must cite a numbered excerpt for every claim; answers carry report title, page number and NTRS link.
* Ran a retrieval ablation on a 40-question answerable golden set: BM25 page hit@5 **0.950** (MRR@10 0.849), dense 0.875, hybrid RRF 0.900, cross-encoder replacing the ranking dropped to 0.800, so I fused the reranker as a third RRF signal instead, recovering page hit@5 **0.925** and doc hit@5 **1.000**.
* Designed a calibrated refusal gate (cross-encoder threshold **-6.10**, fit on a separate 16-question calibration set): on the 26 golden questions scored end-to-end, **8/8 unanswerable questions were declined** without an LLM call, refusal accuracy **96.2%** (25/26), with 1 of 18 answerable questions wrongly declined.
* Measured **100% page-level citation accuracy (17/17 answered questions cite the gold report page)**; Ragas non-LLM context precision **0.786** and recall **0.690** (n=18).
* Made the pipeline survive free-tier API limits: a 5-provider fallback chain (Groq gpt-oss-20b / 120b / qwen3.8-27b / gpt-oss-safeguard-20b, NVIDIA Build), quota-aware retries that shrink `max_tokens` to fit a 2,000-token/min cap, an on-disk response cache, and resumable evaluation files; the model that answered is logged per question.
* Shipped it as a FastAPI service with a web UI and CLI, Docker, and CI (ruff, **61 pytest tests** with no network or models, and an offline eval gate over the committed results).

Latency (shared 2-vCPU sandbox, 1 torch thread, other jobs running): end-to-end p50 **5.5 s** / p95 **7.9 s** over the 26 scored questions, of which retrieval + rerank p50 is 5.0 s.

Not claimed: the end-to-end run covers **26 of the 50** golden questions (all LLM daily quotas ran out); LLM-judged faithfulness was not run.
