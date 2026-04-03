FROM python:3.11-slim

WORKDIR /app

# Install system Chromium for ARM compatibility (Playwright's bundled binaries are x86-only on Linux)
RUN apt-get update && \
    apt-get install -y --no-install-recommends chromium && \
    rm -rf /var/lib/apt/lists/*

ENV PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium

COPY pyproject.toml .
COPY services/ services/

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "services.api.runner"]
