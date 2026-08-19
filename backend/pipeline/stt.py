import logging
import tempfile
import os
import subprocess
from backend.config import settings
from backend.pipeline.saaras_client import transcribe_audio, SaarasAPIError
from backend.pipeline.language import normalize_language

logger = logging.getLogger(__name__)

def convert_webm_to_wav(webm_bytes: bytes) -> bytes:
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        logger.warning("imageio_ffmpeg not installed. Skipping local conversion.")
        return webm_bytes
        
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f_in:
        f_in.write(webm_bytes)
        in_path = f_in.name
        
    out_path = in_path + ".wav"
    try:
        subprocess.run([
            ffmpeg_path, "-y", "-i", in_path,
            "-vn", "-ar", "16000", "-ac", "1", "-f", "wav", out_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        with open(out_path, "rb") as f_out:
            return f_out.read()
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg conversion failed: {e}")
        return webm_bytes  # fallback to original
    finally:
        if os.path.exists(in_path): os.remove(in_path)
        if os.path.exists(out_path): os.remove(out_path)

class STTServiceError(Exception):
    def __init__(self, message, status_code=500):
        super().__init__(message)
        self.status_code = status_code

class STTService:
    def __init__(self):
        self.supported_extensions = {".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a"}
    
    def _validate_audio(self, audio_bytes: bytes, filename: str):
        max_size_bytes = settings.MAX_AUDIO_MB * 1024 * 1024
        if len(audio_bytes) > max_size_bytes:
            raise STTServiceError(f"Audio file exceeds maximum size of {settings.MAX_AUDIO_MB} MB", status_code=413)
        
        ext = ""
        if "." in filename:
            ext = filename[filename.rfind("."):].lower()
            
        if ext and ext not in self.supported_extensions:
            logger.warning(f"Audio extension {ext} not explicitly in supported list, but passing through to Saaras.")
            


    def transcribe(self, audio_bytes: bytes, filename: str) -> dict:
        """
        Validates audio and triggers Saaras API.
        Returns a normalized dict with 'text' and 'language'.
        """
        if not settings.SAARAS_ENABLED:
            raise STTServiceError("Voice functionality is disabled.", status_code=503)
            
        self._validate_audio(audio_bytes, filename)
        
        if filename.endswith(".webm"):
            audio_bytes = convert_webm_to_wav(audio_bytes)
            filename = filename.replace(".webm", ".wav")
        
        try:
            res = transcribe_audio(audio_bytes, filename)
        except SaarasAPIError as e:
            raise STTServiceError(str(e), status_code=e.status_code or 502)
            
        text = res.get("transcript", "").strip()
        if not text:
            raise STTServiceError("Transcription resulted in empty text.", status_code=400)
            
        raw_lang = res.get("language_code")
        lang = normalize_language(raw_lang)
        
        return {
            "text": text,
            "language": lang
        }

stt_service = STTService()
