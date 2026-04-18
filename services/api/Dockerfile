FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY pyproject.toml .
COPY services/ services/
COPY scripts/ scripts/

RUN pip install --no-cache-dir .

# Data directory for SQLite
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "-m", "services.api.runner"]
