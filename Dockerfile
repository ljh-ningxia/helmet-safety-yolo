FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HELMET_LOG_JSON=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY web ./web
COPY scripts ./scripts
COPY README.md NOTICE.md ./
RUN pip install --no-cache-dir --no-deps -e .

# Model weights and dataset are NOT baked in: mount them at runtime.
#   docker run -v $PWD/models:/app/models -v $PWD/data:/app/data ...
VOLUME ["/app/models", "/app/data", "/app/experiments"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "-m", "helmet_safety.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
