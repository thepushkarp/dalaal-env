FROM python:3.12-slim AS builder

WORKDIR /app

# Install system deps for building
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Copy project files
COPY pyproject.toml uv.lock ./
COPY __init__.py client.py models.py openenv.yaml ./
COPY server/ ./server/
COPY mock_sites/ ./mock_sites/
COPY README.md ./

# Install dependencies
RUN uv sync --frozen --no-editable || uv sync --no-editable

# Install Playwright Chromium
RUN /app/.venv/bin/playwright install chromium && \
    /app/.venv/bin/playwright install-deps chromium

# ── Runtime stage ────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Install Chromium runtime dependencies + curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libdbus-1-3 libxkbcommon0 \
    libatspi2.0-0 libxcomposite1 libxdamage1 libxfixes3 \
    libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 \
    libwayland-client0 fonts-noto-color-emoji curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces runs as uid 1000)
RUN useradd -m -u 1000 appuser

# Copy venv and app code from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app
COPY --from=builder /root/.cache/ms-playwright /home/appuser/.cache/ms-playwright

# Fix permissions
RUN chown -R appuser:appuser /app /home/appuser/.cache

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PLAYWRIGHT_BROWSERS_PATH="/home/appuser/.cache/ms-playwright"

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
