FROM python:3.12-slim

RUN apt-get update  \
    && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace
ENV PYTHONPATH="${PYTHONPATH}:."

COPY pyproject.toml .
COPY uv.lock .

RUN uv pip install --system -r pyproject.toml
COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "6", "--log-level", "info"]