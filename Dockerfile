FROM python:3.12-slim-bookworm

# Use UV as package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy code and install dependencies
COPY . .
RUN uv venv && uv pip install -r requirements.txt

ENV PATH="/app/.venv/bin:$PATH"

# Copy entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000 8265
ENTRYPOINT ["/app/entrypoint.sh"]
