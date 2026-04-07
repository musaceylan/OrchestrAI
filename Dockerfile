FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY config/ ./config/

# Install dependencies
RUN uv pip install --system --no-cache .

# Create non-root user
RUN useradd -m -u 1000 orchestrai
RUN mkdir -p /tmp/orchestrai/traces /tmp/orchestrai/artifacts \
    && chown -R orchestrai:orchestrai /tmp/orchestrai

USER orchestrai

# Health check — just verify the CLI is importable
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from orchestrai.server.main import serve; print('ok')"

ENTRYPOINT ["orchestrai"]
CMD ["--log-format", "json"]
