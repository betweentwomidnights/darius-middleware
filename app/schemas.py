"""Pydantic schemas for request/response validation."""

from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================================
# Auth Schemas
# ============================================================================


class MockLoginRequest(BaseModel):
    """Mock login request for Phase 1."""

    test_user_id: str


class AuthResponse(BaseModel):
    """Auth response with JWT token."""

    token: str
    user_id: UUID


# ============================================================================
# Credits Schemas
# ============================================================================


class BalanceResponse(BaseModel):
    """User's credit balance."""

    balance_seconds: Decimal
    formatted: str  # MM:SS format


class AddCreditsRequest(BaseModel):
    """Request to add credits (test endpoint for Phase 1)."""

    seconds: Decimal = Field(gt=0)


# ============================================================================
# Jam Session Schemas
# ============================================================================


class WarmupResponse(BaseModel):
    """Response from /jam/warmup."""

    ok: bool
    session_id: str
    warmed: bool
    billing_state: str
    container_warmup_seconds: float


class JamStartResponse(BaseModel):
    """Response from /jam/start."""

    ok: bool
    session_id: str
    status: str


class JamUpdateResponse(BaseModel):
    """Response from /jam/update."""

    ok: bool
    session_id: str


class JamReseedResponse(BaseModel):
    """Response from /jam/reseed."""

    ok: bool
    session_id: str


class JamReseedSpliceResponse(BaseModel):
    """Response from /jam/reseed_splice."""

    ok: bool
    session_id: str
    anchor_bars: float


class ChunkMetadata(BaseModel):
    """Metadata for a jam chunk."""

    bpm: int
    bars: int
    beats_per_bar: int
    sample_rate: int
    channels: int
    total_samples: int
    seconds_per_bar: float
    loop_duration_seconds: float
    guidance_weight: float
    temperature: float
    topk: int


class JamChunk(BaseModel):
    """A single jam chunk with audio and metadata."""

    index: int
    audio_base64: str
    metadata: ChunkMetadata


class JamNextResponse(BaseModel):
    """Response from /jam/next."""

    ok: bool
    chunk: JamChunk


class JamConsumeResponse(BaseModel):
    """Response from /jam/consume."""

    ok: bool


class JamStopResponse(BaseModel):
    """Response from /jam/stop."""

    ok: bool
    status: str


class JamCloseResponse(BaseModel):
    """Response from /jam/close."""

    ok: bool
    status: str
    total_charged: Decimal


class JamStatusResponse(BaseModel):
    """Response from /jam/status."""

    ok: bool
    billing_state: str
    billable_elapsed_seconds: Decimal
    credits_remaining: Decimal
    jam_running: bool
