"""Main FastAPI application."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.billing import billing_loop
from app.database import init_db
from app.modal_client import get_modal_client
from app.routes import auth, credits, jam


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("Initializing database...")
    await init_db()
    print("Database initialized")

    # Start billing loop
    print("Starting billing loop...")
    billing_task = asyncio.create_task(billing_loop())

    yield

    # Shutdown
    print("Shutting down billing loop...")
    billing_task.cancel()
    try:
        await billing_task
    except asyncio.CancelledError:
        pass
    print("Billing loop stopped")

    # Close Modal client
    print("Closing Modal client...")
    client = get_modal_client()
    await client.close()
    print("Modal client closed")


# Create FastAPI app
app = FastAPI(
    title="Darius Middleware",
    description="Middleware for iOS app to Modal-hosted Magenta-RT",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(auth.router)
app.include_router(credits.router)
app.include_router(jam.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "darius-middleware",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
