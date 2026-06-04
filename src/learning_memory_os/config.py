from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    openai_api_key: str
    database_url: str
    log_dir: Path = Field(default=Path("./logs"), alias="LMOS_LOG_DIR")
    default_token_budget: int = Field(default=8000, alias="LMOS_DEFAULT_TOKEN_BUDGET")
    xtrace_api_key: str | None = Field(default=None, alias="XTRACE_API_KEY")
    xtrace_org_id: str | None = Field(default=None, alias="XTRACE_ORG_ID")
    xtrace_base_url: str = Field(
        default="https://api.production.xtrace.ai", alias="XTRACE_BASE_URL"
    )
    # Auth: set COOKIE_SECURE=true in production (HTTPS). Off for local http testing.
    cookie_secure: bool = Field(default=False, alias="COOKIE_SECURE")
    session_ttl_days: int = Field(default=30, alias="SESSION_TTL_DAYS")

    # Billing (Stripe $5 one-time). When unset, billing is disabled and every
    # signed-in user has full access (dev/test behaves as before).
    stripe_secret_key: str | None = Field(default=None, alias="STRIPE_SECRET_KEY")
    stripe_price_id: str | None = Field(default=None, alias="STRIPE_PRICE_ID")
    stripe_webhook_secret: str | None = Field(default=None, alias="STRIPE_WEBHOOK_SECRET")
    app_base_url: str = Field(default="http://localhost:8000", alias="APP_BASE_URL")

    @property
    def billing_enabled(self) -> bool:
        return bool(self.stripe_secret_key and self.stripe_price_id)


def get_settings() -> Settings:
    return Settings()
