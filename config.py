from dataclasses import dataclass
import os


@dataclass
class Settings:
    """Application-wide settings loaded from environment variables.

    Keep this tiny and extensible; other modules should import `settings`.
    """

    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    cache_enabled: bool = os.getenv("CACHE_ENABLED", "1") in ("1", "true", "True")


settings = Settings()
