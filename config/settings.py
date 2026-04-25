from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_base_url: str = "http://localhost:8080/v1"
    llm_model: str = "qwen-3.5-9b"
    llm_api_key: str = "sk-no-key-required"
    llm_temperature: float = 0.8
    llm_max_tokens: int = 2048

    # Memory
    max_context_messages: int = 20
    db_path: str = "./data/eva.db"
    max_memories_in_context: int = 10
    memory_candidate_pool: int = 50

    # Persona
    default_persona_path: str = "./config/default_persona.yaml"
    personas_dir: str = "./personas"


settings = Settings()
