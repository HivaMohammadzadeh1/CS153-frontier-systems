from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    openai_api_key: str
    database_url: str
    # Chat tutor model — Haiku is ~2x faster than Sonnet (and far faster than Opus)
    # and stays strong for tutoring, so the interactive chat feels snappy. Set
    # LMOS_TUTOR_MODEL=claude-sonnet-4-6 for more technical depth at the cost of speed.
    tutor_model: str = Field(default="claude-haiku-4-5-20251001", alias="LMOS_TUTOR_MODEL")
    # Forced "show your reasoning" pass before the answer. It's a nice thought-process
    # box but adds a few seconds of latency per turn; off by default for responsiveness.
    tutor_reasoning: bool = Field(default=False, alias="LMOS_TUTOR_REASONING")
    # Stronger model the chat auto-escalates to for depth-heavy questions (quantitative
    # reasoning, "why", tradeoffs). Keeps the fast model for everything else.
    tutor_model_deep: str = Field(default="claude-sonnet-4-6", alias="LMOS_TUTOR_MODEL_DEEP")
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


def get_settings() -> Settings:
    return Settings()
