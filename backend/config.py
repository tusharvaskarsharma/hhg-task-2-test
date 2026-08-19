import os


def parse_bool(value: str | None, default: bool = False) -> bool:
    """Robustly parse a string env-var to bool. Avoids the bool('False') == True trap."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    # Set default path to point to the local hhg_rag_artifacts directory in the project root
    _base_dir = os.path.dirname(os.path.dirname(__file__))
    HHG_ARTIFACT_DIR = os.getenv("HHG_ARTIFACT_DIR", os.path.join(_base_dir, "hhg_rag_artifacts"))

    # HNSW — support both HHG_HNSW_EF_SEARCH (canonical) and legacy HNSW_EF_SEARCH
    HNSW_EF_SEARCH = int(
        os.getenv("HHG_HNSW_EF_SEARCH") or os.getenv("HNSW_EF_SEARCH", "32")
    )

    # ONNX thread tuning
    ONNX_INTRA_THREADS = int(os.getenv("HHG_ONNX_INTRA_THREADS", "0"))  # 0 = onnxruntime default
    ONNX_INTER_THREADS = int(os.getenv("HHG_ONNX_INTER_THREADS", "0"))

    # Cache
    CACHE_ENABLED = parse_bool(
        os.getenv("HHG_CACHE_ENABLED") or os.getenv("CACHE_ENABLED"), default=True
    )
    CACHE_MAX_SIZE = int(
        os.getenv("HHG_CACHE_MAX_SIZE") or os.getenv("CACHE_MAX_SIZE", "1000")
    )
    CACHE_TTL_SECONDS = int(
        os.getenv("HHG_CACHE_TTL_SECONDS") or os.getenv("CACHE_TTL_SECONDS", "3600")
    )
    CACHE_VERSION = os.getenv("HHG_CACHE_VERSION") or os.getenv("CACHE_VERSION", "v1")

    # API Settings
    CORS_ORIGINS = os.getenv("HHG_CORS_ORIGINS", "").split(",") if os.getenv("HHG_CORS_ORIGINS") else []
    DEBUG = parse_bool(os.getenv("HHG_DEBUG"), default=False)
    LOG_LEVEL = os.getenv("HHG_LOG_LEVEL", "INFO").upper()

    # Saaras STT Settings
    SAARAS_ENABLED = parse_bool(os.getenv("HHG_SAARAS_ENABLED"), default=False)
    SAARAS_API_KEY = os.getenv("HHG_SAARAS_API_KEY", "")
    SAARAS_BASE_URL = os.getenv("HHG_SAARAS_BASE_URL", "https://api.sarvam.ai/speech-to-text")
    SAARAS_MODEL = os.getenv("HHG_SAARAS_MODEL", "saaras:v3")
    SAARAS_TIMEOUT_SECONDS = int(os.getenv("HHG_SAARAS_TIMEOUT_SECONDS", "30"))
    MAX_AUDIO_MB = int(os.getenv("HHG_MAX_AUDIO_MB", "10"))

    # SLM Settings
    SLM_ENABLED = parse_bool(os.getenv("HHG_SLM_ENABLED"), default=False)
    SLM_PROVIDER = os.getenv("HHG_SLM_PROVIDER", "groq")
    SLM_BASE_URL = os.getenv("HHG_SLM_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
    SLM_MODEL = os.getenv("HHG_SLM_MODEL", "llama-3.1-8b-instant")
    SLM_API_KEY = os.getenv("HHG_SLM_API_KEY", "")
    SLM_TIMEOUT_SECONDS = int(os.getenv("HHG_SLM_TIMEOUT_SECONDS", "10"))
    SLM_MAX_TOKENS = int(os.getenv("HHG_SLM_MAX_TOKENS", "256"))
    SLM_TEMPERATURE = float(os.getenv("HHG_SLM_TEMPERATURE", "0.0"))

    # Grounding Settings
    GROUNDING_TOP_K = int(os.getenv("HHG_GROUNDING_TOP_K", "5"))
    MAX_CONTEXT_CHARS = int(os.getenv("HHG_MAX_CONTEXT_CHARS", "12000"))

    @property
    def manifest_path(self):
        return os.path.join(self.HHG_ARTIFACT_DIR, "config.json")

    @property
    def bm25_dir(self):
        return os.path.join(self.HHG_ARTIFACT_DIR, "bm25")

    @property
    def hnsw_dir(self):
        return os.path.join(self.HHG_ARTIFACT_DIR, "hnsw")

    @property
    def metadata_path(self):
        return os.path.join(self.HHG_ARTIFACT_DIR, "metadata", "passage_metadata.parquet")

    @property
    def onnx_path(self):
        return os.path.join(self.HHG_ARTIFACT_DIR, "embedding", "onnx_int8", "onnx", "model_quint8_avx2.onnx")

    @property
    def validation_summary_path(self):
        return os.path.join(self.HHG_ARTIFACT_DIR, "validation_report.json")


settings = Settings()
