"""Authentication routes."""

from decimal import Decimal

from fastapi import APIRouter
from sqlalchemy import select

from app.auth import create_access_token
from app.dependencies import DBSession
from app.models import Credit, User
from app.schemas import AuthResponse, MockLoginRequest

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/mock-login", response_model=AuthResponse)
async def mock_login(request: MockLoginRequest, db: DBSession) -> AuthResponse:
    """
    Mock login endpoint for Phase 1 development.

    Creates a test user if it doesn't exist and returns a JWT token.
    """
    # Check if user already exists
    result = await db.execute(
        select(User).where(User.apple_user_id == request.test_user_id)
    )
    user = result.scalar_one_or_none()

    if user is None:
        # Create new user
        user = User(
            apple_user_id=request.test_user_id,
            email=f"{request.test_user_id}@test.local",
        )
        db.add(user)
        await db.flush()  # Flush to get user.id

        # Create credit record with initial balance (600 seconds = 10 minutes)
        credit = Credit(
            user_id=user.id,
            balance_seconds=Decimal("600.00"),
        )
        db.add(credit)
        await db.commit()
        await db.refresh(user)
    else:
        # User exists, just refresh
        await db.refresh(user)

    # Create JWT token
    token = create_access_token(user.id)

    return AuthResponse(
        token=token,
        user_id=user.id,
    )
