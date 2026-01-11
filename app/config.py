"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "sqlite+aiosqlite:///./darius_middleware.db"

    # JWT
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 hours

    # Modal API (Phase 2)
    modal_api_url: str | None = None
    modal_api_key: str | None = None

    # Billing settings
    billing_poll_interval_seconds: float = 5.0
    minimum_credits_to_warmup: float = 60.0
    scaledown_window_seconds: int = 600  # 10 minutes

    # Development
    debug: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Global settings instance
settings = Settings()
