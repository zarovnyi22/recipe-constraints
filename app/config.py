from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5434/recipe"
    llm_provider: Literal["gemini", "groq"] = "gemini"
    # Empty = no fallback.
    llm_fallback_provider: Literal["", "gemini", "groq"] = ""
    gemini_api_key: str = ""
    llm_model: str = "gemini-3.5-flash-lite"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_reasoning_effort: str = "low"  # gpt-oss only: low | medium | high
    llm_timeout_seconds: float = 45.0  # one HTTP attempt
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def check_fallback(self) -> "Settings":
        if self.llm_fallback_provider == self.llm_provider:
            raise ValueError("LLM_FALLBACK_PROVIDER must differ from LLM_PROVIDER (or be empty)")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
