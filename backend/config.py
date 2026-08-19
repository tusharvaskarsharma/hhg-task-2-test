import os

class Settings:
    # Set default path to point to the local hhg_rag_artifacts directory in the project root
    _base_dir = os.path.dirname(os.path.dirname(__file__))
    HHG_ARTIFACT_DIR = os.getenv("HHG_ARTIFACT_DIR", os.path.join(_base_dir, "hhg_rag_artifacts"))
    HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "64"))
    
    CACHE_ENABLED = bool(os.getenv("CACHE_ENABLED", "True") == "True")
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    CACHE_VERSION = os.getenv("CACHE_VERSION", "v1")

    # API Settings
    CORS_ORIGINS = os.getenv("HHG_CORS_ORIGINS", "").split(",") if os.getenv("HHG_CORS_ORIGINS") else []
    DEBUG = bool(os.getenv("HHG_DEBUG", "False").lower() in ("true", "1", "yes"))
    LOG_LEVEL = os.getenv("HHG_LOG_LEVEL", "INFO").upper()

    # Saaras STT Settings
    SAARAS_ENABLED = bool(os.getenv("HHG_SAARAS_ENABLED", "False").lower() in ("true", "1", "yes"))
    SAARAS_API_KEY = os.getenv("HHG_SAARAS_API_KEY", "")
    SAARAS_BASE_URL = os.getenv("HHG_SAARAS_BASE_URL", "https://api.sarvam.ai/speech-to-text")
    SAARAS_MODEL = os.getenv("HHG_SAARAS_MODEL", "saaras:v3")
    SAARAS_TIMEOUT_SECONDS = int(os.getenv("HHG_SAARAS_TIMEOUT_SECONDS", "30"))
    MAX_AUDIO_MB = int(os.getenv("HHG_MAX_AUDIO_MB", "10"))

    # SLM Settings
    SLM_ENABLED = bool(os.getenv("HHG_SLM_ENABLED", "False").lower() in ("true", "1", "yes"))
    SLM_PROVIDER = os.getenv("HHG_SLM_PROVIDER", "generic")
    SLM_BASE_URL = os.getenv("HHG_SLM_BASE_URL", "http://localhost:8080/v1")
    SLM_MODEL = os.getenv("HHG_SLM_MODEL", "default")
    SLM_API_KEY = os.getenv("HHG_SLM_API_KEY", "")
    SLM_TIMEOUT_SECONDS = int(os.getenv("HHG_SLM_TIMEOUT_SECONDS", "30"))
    SLM_MAX_TOKENS = int(os.getenv("HHG_SLM_MAX_TOKENS", "1024"))
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
