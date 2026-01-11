"""Jam session routes (proxy to Modal)."""

import uuid
from datetime import datetime, timezone
from decimal import Decimal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlalchemy import select

from app.config import settings
from app.dependencies import CurrentUser, DBSession
from app.modal_client import get_modal_client
from app.models import Credit, JamSession
from app.schemas import (
    JamCloseResponse,
    JamConsumeResponse,
    JamNextResponse,
    JamReseedResponse,
    JamReseedSpliceResponse,
    JamStartResponse,
    JamStatusResponse,
    JamStopResponse,
    JamUpdateResponse,
    WarmupResponse,
)

router = APIRouter(prefix="/jam", tags=["jam"])


@router.post("/warmup", response_model=WarmupResponse)
async def warmup(user: CurrentUser, db: DBSession) -> WarmupResponse:
    """
    Warmup a new jam session container.

    Checks that user has minimum credits required, then creates a session
    and proxies to Modal /warmup.
    """
    # Check credit balance
    result = await db.execute(
        select(Credit).where(Credit.user_id == user.id)
    )
    credit = result.scalar_one_or_none()

    if not credit or credit.balance_seconds < settings.minimum_credits_to_warmup:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient credits. Minimum {settings.minimum_credits_to_warmup} seconds required.",
        )

    # Generate session ID
    session_id = str(uuid.uuid4())

    # Create jam session record
    jam_session = JamSession(
        user_id=user.id,
        modal_session_id=session_id,
        billing_state="warming",
    )
    db.add(jam_session)
    await db.commit()

    # Call Modal warmup
    modal = get_modal_client()
    response = await modal.warmup(session_id)

    # Update session to active state
    if response.get("warmed"):
        jam_session.billing_state = "active"
        jam_session.billable_start_ts = datetime.now(timezone.utc)
        db.add(jam_session)
        await db.commit()

    return WarmupResponse(
        ok=response.get("ok", True),
        session_id=session_id,
        warmed=response.get("warmed", False),
        billing_state=jam_session.billing_state,
        container_warmup_seconds=response.get("container_warmup_seconds", 0.0),
    )


@router.post("/start", response_model=JamStartResponse)
async def jam_start(
    user: CurrentUser,
    db: DBSession,
    session_id: str = Form(...),
    loop_audio: UploadFile = File(...),
    bpm: int = Form(120),
    bars_per_chunk: int = Form(4),
    beats_per_bar: int = Form(4),
    styles: str = Form(""),
    style_weights: str = Form(""),
    loop_weight: float = Form(1.0),
    loudness_mode: str = Form("auto"),
    loudness_headroom_db: float = Form(1.0),
    guidance_weight: float = Form(1.1),
    temperature: float = Form(1.1),
    topk: int = Form(40),
) -> JamStartResponse:
    """
    Start jamming with a loop.

    Proxies to Modal /jam/start with the provided audio loop and parameters.
    """
    # Verify session belongs to user
    result = await db.execute(
        select(JamSession).where(
            JamSession.modal_session_id == session_id,
            JamSession.user_id == user.id,
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.billing_state != "active":
        raise HTTPException(status_code=400, detail="Session not active")

    # Read audio file
    audio_bytes = await loop_audio.read()

    # Call Modal
    modal = get_modal_client()
    response = await modal.jam_start(
        session_id=session_id,
        loop_audio=audio_bytes,
        bpm=bpm,
        bars_per_chunk=bars_per_chunk,
    )

    return JamStartResponse(
        ok=response.get("ok", True),
        session_id=session_id,
        status=response.get("status", "started"),
    )


@router.post("/update", response_model=JamUpdateResponse)
async def jam_update(
    user: CurrentUser,
    db: DBSession,
    session_id: str = Form(...),
    styles: str = Form(""),
    style_weights: str = Form(""),
    loop_weight: float | None = Form(None),
    guidance_weight: float | None = Form(None),
    temperature: float | None = Form(None),
    topk: int | None = Form(None),
) -> JamUpdateResponse:
    """Hot swap generation parameters mid-jam."""

    result = await db.execute(
        select(JamSession).where(
            JamSession.modal_session_id == session_id,
            JamSession.user_id == user.id,
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    modal = get_modal_client()
    response = await modal.jam_update(
        session_id=session_id,
        styles=styles,
        style_weights=style_weights,
        loop_weight=loop_weight,
        guidance_weight=guidance_weight,
        temperature=temperature,
        topk=topk,
    )

    if not response.get("ok", False):
        raise HTTPException(status_code=400, detail=response.get("error", "Update failed"))

    return JamUpdateResponse(
        ok=True,
        session_id=session_id,
    )


@router.post("/reseed", response_model=JamReseedResponse)
async def jam_reseed(
    user: CurrentUser,
    db: DBSession,
    session_id: str = Form(...),
    loop_audio: UploadFile | None = File(None),
) -> JamReseedResponse:
    """Reseed the jam session with optional loop audio."""

    result = await db.execute(
        select(JamSession).where(
            JamSession.modal_session_id == session_id,
            JamSession.user_id == user.id,
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    loop_audio_bytes = await loop_audio.read() if loop_audio else None

    modal = get_modal_client()
    response = await modal.jam_reseed(session_id, loop_audio_bytes)

    if not response.get("ok", False):
        raise HTTPException(status_code=400, detail=response.get("error", "Reseed failed"))

    return JamReseedResponse(
        ok=True,
        session_id=session_id,
    )


@router.post("/reseed_splice", response_model=JamReseedSpliceResponse)
async def jam_reseed_splice(
    user: CurrentUser,
    db: DBSession,
    session_id: str = Form(...),
    anchor_bars: float = Form(2.0),
    combined_audio: UploadFile | None = File(None),
) -> JamReseedSpliceResponse:
    """Splice-based reseed with optional combined audio payload."""

    result = await db.execute(
        select(JamSession).where(
            JamSession.modal_session_id == session_id,
            JamSession.user_id == user.id,
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    combined_audio_bytes = await combined_audio.read() if combined_audio else None

    modal = get_modal_client()
    response = await modal.jam_reseed_splice(
        session_id=session_id,
        anchor_bars=anchor_bars,
        combined_audio_bytes=combined_audio_bytes,
    )

    if not response.get("ok", False):
        raise HTTPException(status_code=400, detail=response.get("error", "Reseed splice failed"))

    return JamReseedSpliceResponse(
        ok=True,
        session_id=session_id,
        anchor_bars=response.get("anchor_bars", anchor_bars),
    )


@router.get("/next", response_model=JamNextResponse)
async def jam_next(
    user: CurrentUser,
    db: DBSession,
    session_id: str,
    timeout_seconds: int = 10,
) -> JamNextResponse:
    """
    Get the next jam chunk.

    Polls Modal /jam/next for the next available audio chunk.
    """
    # Verify session belongs to user
    result = await db.execute(
        select(JamSession).where(
            JamSession.modal_session_id == session_id,
            JamSession.user_id == user.id,
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Call Modal
    modal = get_modal_client()
    response = await modal.jam_next(session_id, timeout_seconds)

    if not response.get("ok"):
        raise HTTPException(status_code=500, detail=response.get("error", "Unknown error"))

    return JamNextResponse(
        ok=True,
        chunk=response["chunk"],
    )


@router.post("/consume", response_model=JamConsumeResponse)
async def jam_consume(
    user: CurrentUser,
    db: DBSession,
    session_id: str = Form(...),
    chunk_index: int = Form(...),
) -> JamConsumeResponse:
    """
    Mark a chunk as consumed.

    Tells Modal that the chunk has been received and can be removed from the queue.
    """
    # Verify session belongs to user
    result = await db.execute(
        select(JamSession).where(
            JamSession.modal_session_id == session_id,
            JamSession.user_id == user.id,
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Call Modal
    modal = get_modal_client()
    response = await modal.jam_consume(session_id, chunk_index)

    return JamConsumeResponse(ok=response.get("ok", True))


@router.post("/stop", response_model=JamStopResponse)
async def jam_stop(
    user: CurrentUser,
    db: DBSession,
    session_id: str = Form(...),
) -> JamStopResponse:
    """
    Stop the current jam (but keep session warm).

    The container remains billable until /jam/close is called.
    """
    # Verify session belongs to user
    result = await db.execute(
        select(JamSession).where(
            JamSession.modal_session_id == session_id,
            JamSession.user_id == user.id,
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Call Modal
    modal = get_modal_client()
    response = await modal.jam_stop(session_id)

    return JamStopResponse(
        ok=response.get("ok", True),
        status=response.get("status", "stopped"),
    )


@router.post("/close", response_model=JamCloseResponse)
async def jam_close(
    user: CurrentUser,
    db: DBSession,
    session_id: str = Form(...),
) -> JamCloseResponse:
    """
    Close the jam session and finalize billing.

    This stops billing immediately and marks the session as closed.
    """
    # Verify session belongs to user
    result = await db.execute(
        select(JamSession).where(
            JamSession.modal_session_id == session_id,
            JamSession.user_id == user.id,
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Call Modal
    modal = get_modal_client()
    response = await modal.session_close(session_id)

    # Update session
    session.billing_state = "closed"
    session.closed_at = datetime.now(timezone.utc)
    db.add(session)
    await db.commit()

    return JamCloseResponse(
        ok=response.get("ok", True),
        status="closed",
        total_charged=session.credits_charged,
    )


@router.get("/status", response_model=JamStatusResponse)
async def jam_status(
    user: CurrentUser,
    db: DBSession,
    session_id: str,
) -> JamStatusResponse:
    """
    Get the current status of a jam session.

    Returns billing state, elapsed time, credits remaining, and jam running status.
    """
    # Verify session belongs to user
    result = await db.execute(
        select(JamSession).where(
            JamSession.modal_session_id == session_id,
            JamSession.user_id == user.id,
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get user's credit balance
    credit_result = await db.execute(
        select(Credit).where(Credit.user_id == user.id)
    )
    credit = credit_result.scalar_one()

    # Call Modal for current status
    modal = get_modal_client()
    response = await modal.status_peek(session_id)

    billable_elapsed = Decimal(str(response.get("billable_elapsed_seconds", 0)))

    return JamStatusResponse(
        ok=True,
        billing_state=response.get("billing_state", "unknown"),
        billable_elapsed_seconds=billable_elapsed,
        credits_remaining=credit.balance_seconds,
        jam_running=response.get("jam_running", False),
    )
