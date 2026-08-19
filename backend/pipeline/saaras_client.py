import requests
import logging
from backend.config import settings

logger = logging.getLogger(__name__)

class SaarasAPIError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code

def transcribe_audio(audio_bytes: bytes, filename: str) -> dict:
    """
    Sends an audio payload to the Sarvam AI Saaras REST API.
    Returns the JSON response dict containing 'transcript' and 'language_code'.
    """
    if not settings.SAARAS_ENABLED:
        raise SaarasAPIError("Saaras STT is disabled in configuration.")
    if not settings.SAARAS_API_KEY:
        raise SaarasAPIError("Saaras API Key is missing.")

    url = settings.SAARAS_BASE_URL
    
    headers = {
        "api-subscription-key": settings.SAARAS_API_KEY
    }
    
    content_type = "audio/webm" if filename.endswith(".webm") else "audio/wav"
    
    files = {
        "file": (filename, audio_bytes, content_type)
    }
    
    data = {
        "model": settings.SAARAS_MODEL,
        "mode": "transcribe"
    }
    
    try:
        response = requests.post(
            url, 
            headers=headers, 
            files=files, 
            data=data, 
            timeout=settings.SAARAS_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.Timeout as e:
        logger.error(
            f"Saaras STT request timed out. URL: {url} | "
            f"Audio Size: {len(audio_bytes)} | MIME: {content_type} | "
            f"Exception: {type(e).__name__}"
        )
        raise SaarasAPIError("Upstream STT timeout", status_code=504)
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        response_body = e.response.text
        logger.error(
            f"Saaras HTTP Error | URL: {url} | "
            f"Status: {status_code} | "
            f"Response Body: {response_body} | "
            f"MIME Type: {content_type} | "
            f"Audio Size: {len(audio_bytes)}"
        )
        raise SaarasAPIError(f"Upstream STT failed: {response_body}", status_code=status_code)
    except ValueError as e:
        logger.error(
            f"Failed to parse Saaras response as JSON. URL: {url} | "
            f"Audio Size: {len(audio_bytes)} | MIME: {content_type} | "
            f"Exception: {type(e).__name__}"
        )
        raise SaarasAPIError("Invalid response from upstream STT", status_code=502)
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Saaras Request Error: {str(e)}. URL: {url} | "
            f"Audio Size: {len(audio_bytes)} | MIME: {content_type} | "
            f"Exception: {type(e).__name__}"
        )
        raise SaarasAPIError("Upstream STT connection failed", status_code=502)
