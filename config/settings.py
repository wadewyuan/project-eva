from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (main model — handles chat, memory extraction)
    llm_base_url: str = "http://localhost:8080/v1"
    llm_model: str = "qwen-3.5-9b"
    llm_api_key: str = "sk-no-...ired"
    llm_temperature: float = 0.8
    llm_max_tokens: int = 2048

    # LLM (small model — tone detection only)
    # If not set, falls back to the main LLM settings above.
    llm_small_base_url: str | None = None
    llm_small_model: str | None = None
    llm_small_api_key: str | None = None

    # Memory
    max_context_messages: int = 20
    db_path: str = "./data/eva.db"
    max_memories_in_context: int = 10
    memory_candidate_pool: int = 50

    # Voice
    asr_max_workers: int = 1

    # TTS
    tts_provider: Literal["edge", "voxcpm"] = "edge"
    tts_voxcpm_model_source: Literal["hf", "modelscope"] = "hf"
    tts_voxcpm_cfg_value: float = 2.0
    tts_voxcpm_inference_timesteps: int = 10
    tts_voxcpm_voice: str | None = None
    # Voice cloning: global default reference audio + transcript. Used for any
    # persona that doesn't define its own `voxcpm_ref_wav`. Without these,
    # VoxCPM samples a random speaker per call (each sentence sounds different).
    tts_voxcpm_ref_wav: str | None = None
    tts_voxcpm_prompt_text: str | None = None
    tts_voxcpm_hf_endpoint: str = "https://hf-mirror.com"
    tts_voxcpm_load_denoiser: bool = False
    tts_voxcpm_cache_dir: str | None = None  # defaults to ~/.cache/huggingface/hub or modelscope

    # Persona
    default_persona_path: str = "./config/default_persona.yaml"
    personas_dir: str = "./personas"


settings = Settings()
