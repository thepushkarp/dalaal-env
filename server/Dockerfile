ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY . /app/env

WORKDIR /app/env

# Install deps with pip into system Python
RUN pip install --no-cache-dir -e ".[server]" 2>/dev/null || pip install --no-cache-dir -e .

# Install Playwright and Chromium browser
RUN playwright install chromium && \
    playwright install-deps chromium

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

# Install Chromium runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libdbus-1-3 libxkbcommon0 \
    libatspi2.0-0 libxcomposite1 libxdamage1 libxfixes3 \
    libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 \
    libwayland-client0 fonts-noto-color-emoji curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages and app code from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/env /app/env
COPY --from=builder /root/.cache/ms-playwright /root/.cache/ms-playwright

ENV PYTHONPATH="/app/env:$PYTHONPATH"

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

WORKDIR /app/env
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
