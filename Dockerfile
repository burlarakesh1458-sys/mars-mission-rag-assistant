# CPU-only image: FastAPI service + web UI. Corpus PDFs are not baked in; the image ships
# the committed chunks.jsonl and builds the Chroma index + downloads the two small models
# at build time so the container starts offline.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false ANONYMIZED_TELEMETRY=False \
    HF_HOME=/app/.hf PYTHONPATH=/app/src

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data/manifest.jsonl data/chunks.jsonl ./data/
COPY eval ./eval
COPY reports ./reports

# bake models + vector index into the image (about 2-10 min on CPU)
RUN python -m marsrag.cli index \
 && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
ENV HF_HUB_OFFLINE=1

RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
# GROQ_API_KEY (and optionally NVIDIA_API_KEY) must be passed at run time:
#   docker run -p 8000:8000 -e GROQ_API_KEY=... marsrag
CMD ["uvicorn", "marsrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
