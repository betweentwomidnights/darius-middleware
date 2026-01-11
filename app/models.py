"""SQLAlchemy ORM models for users, credits, and jam sessions."""

import uuid
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import DECIMAL, ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class User(Base):
    """User account linked to Apple ID."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    apple_user_id: Mapped[str] = mapped_column(Text, unique=True, index=True)
    email: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    credit: Mapped["Credit"] = relationship(back_populates="user", uselist=False)
    jam_sessions: Mapped[list["JamSession"]] = relationship(back_populates="user")


class Credit(Base):
    """Credit balance for a user (stored as seconds of jam time)."""

    __tablename__ = "credits"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    balance_seconds: Mapped[Decimal] = mapped_column(
        DECIMAL(12, 2),
        default=Decimal("0.00"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="credit")


class JamSession(Base):
    """A billing session for Modal-hosted Magenta-RT."""

    __tablename__ = "jam_sessions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    modal_session_id: Mapped[str] = mapped_column(Text, unique=True, index=True)

    # Billing state: "warming" | "active" | "closed"
    billing_state: Mapped[str] = mapped_column(Text, default="warming")

    # Timestamps
    billable_start_ts: Mapped[datetime | None] = mapped_column(nullable=True)
    billable_end_ts: Mapped[datetime | None] = mapped_column(nullable=True)

    # Credits charged so far
    credits_charged: Mapped[Decimal] = mapped_column(
        DECIMAL(12, 2),
        default=Decimal("0.00"),
    )

    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
    )
    closed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="jam_sessions")
