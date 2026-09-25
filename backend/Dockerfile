FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/models \
    FINSIGHT_SEED_DIR=/app/data/seed \
    FINSIGHT_DATA_DIR=/tmp/finsight \
    PORT=7860

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the ONNX models into the image so a cold start never downloads them.
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/models'); \
TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2', cache_dir='/app/models')"

COPY app ./app
COPY eval/results.json ./eval/results.json
COPY data/seed ./data/seed
# Hugging Face Spaces runs the container as uid 1000; keep everything it touches writable.
RUN mkdir -p /tmp/finsight && chmod -R a+rwX /app/models /tmp/finsight

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
