# ULD Forecasting System
# Multi-stage Docker build for production deployment

# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml ./
COPY requirements*.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install production dependencies
RUN uv pip install --no-cache -r requirements.txt 2>/dev/null || \
    pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    pydantic>=2.5.0 \
    httpx>=0.25.0 \
    numpy>=1.26.0 \
    scipy>=1.11.0 \
    pandas>=2.1.0 \
    sqlalchemy>=2.0.0 \
    aiosqlite>=0.19.0 \
    asyncpg>=0.29.0 \
    python-dotenv>=1.0.0


# ============================================================
# Stage 2: Production
# ============================================================
FROM python:3.12-slim as production

WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY demo.py ./

# Create data directory
RUN mkdir -p /app/data && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Run the application
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]


# ============================================================
# Stage 3: Development
# ============================================================
FROM python:3.12-slim as development

WORKDIR /app

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest>=7.4.0 \
    pytest-asyncio>=0.21.0 \
    pytest-cov>=4.1.0 \
    ruff>=0.1.0 \
    mypy>=1.7.0

# Copy all code
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

EXPOSE ${PORT}

# Default: run with reload for development
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
