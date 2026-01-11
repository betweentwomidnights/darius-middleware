"""Credits management routes."""

from decimal import Decimal

from fastapi import APIRouter, HTTPException
from sqlalchemy import select, update

from app.dependencies import CurrentUser, DBSession
from app.models import Credit
from app.schemas import AddCreditsRequest, BalanceResponse

router = APIRouter(prefix="/credits", tags=["credits"])


def format_seconds(seconds: Decimal) -> str:
    """Format seconds as MM:SS."""
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes}:{secs:02d}"


@router.get("/balance", response_model=BalanceResponse)
async def get_balance(user: CurrentUser, db: DBSession) -> BalanceResponse:
    """Get the current user's credit balance."""
    result = await db.execute(
        select(Credit).where(Credit.user_id == user.id)
    )
    credit = result.scalar_one_or_none()

    if credit is None:
        raise HTTPException(status_code=404, detail="Credit record not found")

    return BalanceResponse(
        balance_seconds=credit.balance_seconds,
        formatted=format_seconds(credit.balance_seconds),
    )


@router.post("/add", response_model=BalanceResponse)
async def add_credits(
    request: AddCreditsRequest,
    user: CurrentUser,
    db: DBSession,
) -> BalanceResponse:
    """
    Add credits to the user's balance (test endpoint for Phase 1).

    In Phase 2+, this will validate Apple in-app purchase receipts.
    """
    result = await db.execute(
        update(Credit)
        .where(Credit.user_id == user.id)
        .values(balance_seconds=Credit.balance_seconds + request.seconds)
        .returning(Credit.balance_seconds)
    )
    await db.commit()

    new_balance = result.scalar_one()

    return BalanceResponse(
        balance_seconds=new_balance,
        formatted=format_seconds(new_balance),
    )
