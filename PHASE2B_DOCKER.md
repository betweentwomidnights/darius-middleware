# Phase 2B: Dockerization

## Overview

Create a Dockerfile for the middleware that can be built locally and then integrated into the existing T4 docker-compose network. Postgres will be added to the docker-compose on the VM.

## Task 1: Update requirements.txt

Add `asyncpg` for PostgreSQL support:

```txt
# Database (async SQLite for local dev, Postgres for production)
sqlalchemy[asyncio]>=2.0.25
aiosqlite>=0.19.0
asyncpg>=0.29.0
```

## Task 2: Create Dockerfile

Create `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Task 3: Create .dockerignore

Create `.dockerignore` to keep the image slim:

```
.venv/
venv/
__pycache__/
*.pyc
*.pyo
*.db
*.sqlite
*.log
.env
.git/
.gitignore
*.md
reference/
test_*.py
test_*.sh
.pytest_cache/
.coverage
htmlcov/
```

## Task 4: Verify Database URL Handling

The app should already handle both SQLite and Postgres URLs via the `DATABASE_URL` environment variable. Verify `app/database.py` works with both:

```python
# SQLite (local dev)
DATABASE_URL=sqlite+aiosqlite:///./darius_middleware.db

# Postgres (docker)
DATABASE_URL=postgresql+asyncpg://darius:password@postgres:5432/darius_middleware
```

SQLAlchemy's async engine handles both dialects transparently. No code changes should be needed if you're using `create_async_engine(settings.database_url)`.

## Task 5: Build and Test Locally

```bash
# Build the image
docker build -t darius-middleware:latest .

# Test run (still using SQLite, just verifying the container works)
docker run --rm -p 8000:8000 \
  -e DATABASE_URL=sqlite+aiosqlite:///./test.db \
  -e DARIUS_API_KEY=test-key \
  -e MODAL_API_URL=https://the-collabage-patch--modal-darius-web.modal.run \
  -e JWT_SECRET=test-secret \
  darius-middleware:latest

# In another terminal, test health endpoint
curl http://localhost:8000/health
# Should return: {"status":"healthy"}

# Stop with Ctrl+C
```

## Task 6: Push to GitHub

Once the build succeeds:

```bash
git add Dockerfile .dockerignore requirements.txt
git commit -m "Add Dockerfile for containerization"
git push
```

## Reference: Docker-Compose Integration (for VM)

When you get to the VM, you'll add these to the existing `docker-compose.yml`:

```yaml
services:
  # ... existing services (g4lwebsockets, gpu-queue, etc.) ...

  # NEW: PostgreSQL for darius-middleware
  darius-postgres:
    image: postgres:16-alpine
    restart: unless-stopped
    volumes:
      - darius-postgres-data:/var/lib/postgresql/data
    environment:
      POSTGRES_USER: darius
      POSTGRES_PASSWORD: ${DARIUS_POSTGRES_PASSWORD}
      POSTGRES_DB: darius_middleware
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U darius -d darius_middleware"]
      interval: 5s
      timeout: 5s
      retries: 5

  # NEW: Darius middleware
  darius-middleware:
    build:
      context: ./darius-middleware
      dockerfile: Dockerfile
    restart: unless-stopped
    ports:
      - "8010:8000"
    depends_on:
      darius-postgres:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql+asyncpg://darius:${DARIUS_POSTGRES_PASSWORD}@darius-postgres:5432/darius_middleware
      DARIUS_API_KEY: ${DARIUS_API_KEY}
      MODAL_API_URL: https://the-collabage-patch--modal-darius-web.modal.run
      JWT_SECRET: ${JWT_SECRET}
      BILLING_POLL_INTERVAL_SECONDS: 5.0
      MINIMUM_CREDITS_TO_WARMUP: 60.0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

volumes:
  # ... existing volumes ...
  darius-postgres-data:  # NEW
```

Then add to your `.env` on the VM:
```
DARIUS_POSTGRES_PASSWORD=<generate-secure-password>
DARIUS_API_KEY=<your-modal-api-key>
JWT_SECRET=<generate-secure-secret>
```

## Notes

### Port Selection
Using 8010 for the middleware since your existing services use:
- 8000: g4lwebsockets
- 8002: melodyflow
- 8005: stable-audio
- 8085: gpu-queue

### No GPU Needed
The middleware is CPU-only (just proxying to Modal). No `deploy.resources.reservations` needed.

### Database Initialization
SQLAlchemy's `create_all()` runs on startup and creates tables if they don't exist. For a fresh Postgres, the tables will be created automatically on first run.

### Backup Strategy (for later)
```bash
# Backup
docker-compose exec darius-postgres pg_dump -U darius darius_middleware > backup.sql

# Restore
cat backup.sql | docker-compose exec -T darius-postgres psql -U darius darius_middleware
```