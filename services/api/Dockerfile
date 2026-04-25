FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV ARGUS_MODEL_PATH=/app/checkpoints/best_model.pt

COPY pyproject.toml .
COPY services/ services/
COPY scripts/ scripts/

# Install base + ML dependencies (torch CPU-only to keep image small)
RUN pip install --no-cache-dir ".[ml]" --extra-index-url https://download.pytorch.org/whl/cpu

# Data and checkpoints directories
RUN mkdir -p /app/data /app/checkpoints

EXPOSE 8000

CMD ["python", "-m", "services.api.runner"]
