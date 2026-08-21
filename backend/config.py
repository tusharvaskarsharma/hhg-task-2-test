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
    # Retrieval cache variables fallback to older CACHE_ variables for backwards compatibility
    RETRIEVAL_CACHE_ENABLED = parse_bool(
        os.getenv("HHG_RETRIEVAL_CACHE_ENABLED") or os.getenv("HHG_CACHE_ENABLED") or os.getenv("CACHE_ENABLED"), default=True
    )
    RETRIEVAL_CACHE_MAX_SIZE = int(
        os.getenv("HHG_RETRIEVAL_CACHE_MAX_SIZE") or os.getenv("HHG_CACHE_MAX_SIZE") or os.getenv("CACHE_MAX_SIZE", "1000")
    )
    RETRIEVAL_CACHE_TTL_SECONDS = int(
        os.getenv("HHG_RETRIEVAL_CACHE_TTL_SECONDS") or os.getenv("HHG_CACHE_TTL_SECONDS") or os.getenv("CACHE_TTL_SECONDS", "3600")
    )
    
    # New Response Cache settings
    RESPONSE_CACHE_ENABLED = parse_bool(
        os.getenv("HHG_RESPONSE_CACHE_ENABLED"), default=True
    )
    RESPONSE_CACHE_MAX_SIZE = int(
        os.getenv("HHG_RESPONSE_CACHE_MAX_SIZE", "500")
    )
    RESPONSE_CACHE_TTL_SECONDS = int(
        os.getenv("HHG_RESPONSE_CACHE_TTL_SECONDS", "900")
    )
    
    # Common Cache Settings
    CACHE_SCHEMA_VERSION = os.getenv("HHG_CACHE_SCHEMA_VERSION") or os.getenv("HHG_CACHE_VERSION") or os.getenv("CACHE_VERSION", "v2")

    # API Settings
    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv(
            "HHG_CORS_ORIGINS",
            "http://127.0.0.1:3000,http://localhost:3000",
        ).split(",")
        if origin.strip()
    ]
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
    SLM_MODEL = os.getenv("HHG_SLM_MODEL", "groq/compound-mini")
    SLM_API_KEY = os.getenv("HHG_SLM_API_KEY", "")
    SLM_TIMEOUT_SECONDS = float(os.getenv("HHG_SLM_TIMEOUT_SECONDS", "5.0"))

    SLM_MAX_TOKENS = int(os.getenv("HHG_SLM_MAX_TOKENS", "128"))
    SLM_TEMPERATURE = float(os.getenv("HHG_SLM_TEMPERATURE", "0.0"))
    SLM_COALESCE_TIMEOUT_SECONDS = int(os.getenv("HHG_SLM_COALESCE_TIMEOUT_SECONDS", "30"))

    # Grounding Settings
    GROUNDING_TOP_K = int(os.getenv("HHG_GROUNDING_TOP_K", "3"))
    MAX_CONTEXT_CHARS = int(os.getenv("HHG_MAX_CONTEXT_CHARS", "4000"))

    # Prompt and Grounding Policy Versioning
    PROMPT_VERSION = os.getenv("HHG_PROMPT_VERSION", "v2")
    GROUNDING_VERSION = os.getenv("HHG_GROUNDING_VERSION", "v1")

    # Extractive Settings
    EXTRACTIVE_MIN_QUERY_COVERAGE = float(os.getenv("HHG_EXTRACTIVE_MIN_QUERY_COVERAGE", "0.60"))
    EXTRACTIVE_MIN_SCORE = float(os.getenv("HHG_EXTRACTIVE_MIN_SCORE", "0.20"))
    EXTRACTIVE_REQUIRE_LANGUAGE_MATCH = parse_bool(os.getenv("HHG_EXTRACTIVE_REQUIRE_LANGUAGE_MATCH"), default=True)
    EXTRACTIVE_MAX_SENTENCE_CHARS = int(os.getenv("HHG_EXTRACTIVE_MAX_SENTENCE_CHARS", "500"))

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
