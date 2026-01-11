"""Background billing loop for jam sessions."""

import asyncio
from decimal import Decimal

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import AsyncSessionLocal
from app.modal_client import get_modal_client
from app.models import Credit, JamSession


async def deduct_credits(db: AsyncSession, user_id, amount: Decimal) -> bool:
    """
    Atomically deduct credits from a user's balance.

    Returns True if successful, False if insufficient balance.
    """
    result = await db.execute(
        update(Credit)
        .where(Credit.user_id == user_id)
        .where(Credit.balance_seconds >= amount)
        .values(balance_seconds=Credit.balance_seconds - amount)
    )
    await db.commit()
    return result.rowcount > 0


async def billing_loop():
    """
    Background task that polls active jam sessions and deducts credits.

    Runs every BILLING_POLL_INTERVAL_SECONDS and:
    1. Queries active sessions (billing_state = "active")
    2. Calls Modal /status/peek to get billable_elapsed_seconds
    3. Calculates delta since last charge
    4. Deducts credits from user balance
    5. Updates session.credits_charged
    6. Handles insufficient balance (force close session)
    7. Finalizes sessions that Modal has marked as "closed"
    """
    modal = get_modal_client()

    while True:
        try:
            await asyncio.sleep(settings.billing_poll_interval_seconds)

            async with AsyncSessionLocal() as db:
                # Query active sessions
                result = await db.execute(
                    select(JamSession).where(JamSession.billing_state == "active")
                )
                active_sessions = result.scalars().all()

                for session in active_sessions:
                    try:
                        # Peek status from Modal
                        status = await modal.status_peek(session.modal_session_id)

                        if not status.get("ok"):
                            continue

                        # Calculate elapsed time
                        elapsed = Decimal(str(status.get("billable_elapsed_seconds", 0)))
                        already_charged = session.credits_charged
                        delta = elapsed - already_charged

                        if delta <= 0:
                            # Modal may have already closed or we've billed everything.
                            if status.get("billing_state") == "closed" and session.billing_state != "closed":
                                session.billing_state = "closed"
                                db.add(session)
                                await db.commit()
                            continue

                        # Check if session is closed on Modal side
                        if status.get("billing_state") == "closed":
                            # Deduct final delta
                            success = await deduct_credits(db, session.user_id, delta)
                            if success:
                                session.credits_charged = elapsed
                                session.billing_state = "closed"
                                db.add(session)
                                await db.commit()
                            continue

                        # Try to deduct credits
                        success = await deduct_credits(db, session.user_id, delta)

                        if success:
                            # Update session
                            session.credits_charged += delta
                            db.add(session)
                            await db.commit()
                        else:
                            # Insufficient balance - force close session
                            await modal.session_close(session.modal_session_id)
                            session.billing_state = "closed"
                            db.add(session)
                            await db.commit()

                    except Exception as e:
                        # Log error but continue processing other sessions
                        print(f"Error billing session {session.id}: {e}")
                        continue

        except asyncio.CancelledError:
            print("Billing loop cancelled")
            break
        except Exception as e:
            print(f"Error in billing loop: {e}")
            await asyncio.sleep(1.0)
