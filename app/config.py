"""Application settings, loaded from environment variables / `.env`.

Every key may be empty. `env_ignore_empty=True` means an empty value in .env
falls back to the default below instead of overriding it with "".
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ORBITDESK_ENV_FILE lets tests point at a non-existent file so a real key in .env never leaks in.
    model_config = SettingsConfigDict(
        env_file=os.environ.get("ORBITDESK_ENV_FILE", ROOT_DIR / ".env"),
        env_ignore_empty=True,
        extra="ignore",
    )

    # --- Groq ---
    groq_api_key: str = ""
    groq_chat_model: str = "llama-3.3-70b-versatile"
    groq_small_model: str = "llama-3.1-8b-instant"  # router, rewrite, injection classifier, 1st fallback
    groq_vision_model: str = "qwen/qwen3.8-27b"
    groq_stt_model: str = "whisper-large-v3-turbo"

    # --- Optional second provider (OpenRouter, OpenAI-compatible API) ---
    fallback_provider_key: str = ""
    fallback_base_url: str = "https://openrouter.ai/api/v1"
    fallback_chat_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    fallback_vision_model: str = "qwen/qwen2.5-vl-72b-instruct:free"

    # --- Langfuse ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    hf_token: str = ""

    # --- Behaviour switches ---
    prompt_version: str = "v1"
    router_mode: str = "prompt"  # prompt | finetuned
    router_adapter: str = ""  # HF Hub id or local path of the LoRA adapter (ROUTER_MODE=finetuned)

    # --- Infra ---
    redis_url: str = "redis://localhost:6379/0"
    chroma_url: str = ""  # e.g. http://chroma:8000 ; empty = embedded on-disk Chroma
    queue_mode: str = "rq"  # rq (worker process) | inline (run job in-process, no worker needed)
    data_dir: Path = ROOT_DIR / "data"
    docs_dir: Path = ROOT_DIR / "data" / "docs"
    prompts_dir: Path = ROOT_DIR / "prompts"
    reports_dir: Path = ROOT_DIR / "reports"
    ingest_on_startup: bool = True

    # --- Retrieval ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    vector_backend: str = "chroma"  # chroma | faiss
    chunk_size: int = 300  # tokens (embedding tokenizer); model limit is 512
    chunk_overlap: int = 45
    embed_batch_size: int = 8  # fastest on CPU in reports/embedding_benchmark.json
    candidate_k: int = 20  # per retriever, before fusion
    top_k: int = 5  # after rerank
    use_hybrid: bool = True
    use_rerank: bool = True
    # Cross-encoder logit below which we answer "I couldn't find this". Tuned in Phase 6.
    relevance_threshold: float = -4.0

    # --- Guardrails / reliability ---
    max_question_chars: int = 1000
    rate_limit_per_minute: int = 20
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    agent_max_steps: int = 6
    agent_timeout_s: float = 60.0
    memory_token_budget: int = 1500
    memory_recent_turns: int = 6
    semantic_cache_threshold: float = 0.95  # see reports/semantic_cache_calibration.json
    injection_llm_check: bool = True  # 2nd-layer LLM safety classifier on user input

    @property
    def llm_available(self) -> bool:
        return bool(self.groq_api_key or self.fallback_provider_key)

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "orbitdesk.db"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def faiss_dir(self) -> Path:
        return self.data_dir / "faiss"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()
