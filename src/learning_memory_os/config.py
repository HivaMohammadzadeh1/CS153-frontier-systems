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


def get_settings() -> Settings:
    return Settings()
