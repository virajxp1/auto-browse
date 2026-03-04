FROM mcr.microsoft.com/playwright/python:v1.55.0-noble

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY agent ./agent
COPY auto_browse ./auto_browse
COPY config ./config
COPY scripts ./scripts

RUN chmod +x scripts/run_api.sh \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

CMD ["./scripts/run_api.sh"]
