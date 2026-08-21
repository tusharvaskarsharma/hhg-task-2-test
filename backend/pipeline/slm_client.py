import requests
import logging
from backend.config import settings

logger = logging.getLogger(__name__)

class SLMClientError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code

class SLMClient:
    def __init__(self):
        self.enabled = settings.SLM_ENABLED
        self.provider = settings.SLM_PROVIDER
        self.base_url = settings.SLM_BASE_URL
        self.model = settings.SLM_MODEL
        self.api_key = settings.SLM_API_KEY
        self.timeout = settings.SLM_TIMEOUT_SECONDS
        self.max_tokens = settings.SLM_MAX_TOKENS
        self.temperature = settings.SLM_TEMPERATURE
        
        # Persistent session for TCP connection reuse — no retries on critical path
        self.session = requests.Session()

    def generate(self, prompt: str) -> str:
        if not self.enabled:
            raise SLMClientError("SLM is disabled in configuration.")
            
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        try:
            response = self.session.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=(min(self.timeout, 2), self.timeout)  # (connect_timeout, read_timeout)
            )
            response.raise_for_status()
            
            data = response.json()
            # Standard OpenAI compatible response structure
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")
                return content.strip()
            elif "response" in data:  # fallback for some Ollama-like APIs
                return data["response"].strip()
                
            raise SLMClientError("Malformed response from SLM provider", status_code=502)
            
        except requests.exceptions.Timeout:
            logger.error("SLM request timed out.")
            raise SLMClientError("Upstream SLM timeout", status_code=504)
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            logger.error(f"SLM HTTP Error {status_code}")
            raise SLMClientError("Upstream SLM failed", status_code=502)
        except requests.exceptions.RequestException as e:
            logger.error(f"SLM Request Error: {type(e).__name__}")
            raise SLMClientError("Upstream SLM connection failed", status_code=502)
        except ValueError:
            logger.error("Failed to parse SLM response as JSON.")
            raise SLMClientError("Invalid response from upstream SLM", status_code=502)
            
    def health(self) -> dict:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "model": self.model
        }

slm_client = SLMClient()
