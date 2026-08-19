import pytest
import io
from unittest.mock import patch, MagicMock
from backend.config import settings
from backend.pipeline.stt import stt_service, STTServiceError
from backend.pipeline.saaras_client import SaarasAPIError

def test_stt_disabled(monkeypatch):
    monkeypatch.setattr(settings, "SAARAS_ENABLED", False)
    with pytest.raises(STTServiceError) as exc:
        stt_service.transcribe(b"dummy", "test.wav")
    assert exc.value.status_code == 503

def test_oversized_audio(monkeypatch):
    monkeypatch.setattr(settings, "SAARAS_ENABLED", True)
    monkeypatch.setattr(settings, "MAX_AUDIO_MB", 1) # 1 MB
    
    # Create 2MB of dummy data
    large_audio = b"0" * (2 * 1024 * 1024)
    
    with pytest.raises(STTServiceError) as exc:
        stt_service.transcribe(large_audio, "test.wav")
    assert exc.value.status_code == 413

@patch("backend.pipeline.stt.transcribe_audio")
def test_successful_transcription(mock_transcribe, monkeypatch):
    monkeypatch.setattr(settings, "SAARAS_ENABLED", True)
    
    mock_transcribe.return_value = {
        "transcript": "Hello world",
        "language_code": "en-IN"
    }
    
    res = stt_service.transcribe(b"dummy", "test.wav")
    
    assert res["text"] == "Hello world"
    assert res["language"] == "en"

@patch("backend.pipeline.stt.transcribe_audio")
def test_empty_transcription(mock_transcribe, monkeypatch):
    monkeypatch.setattr(settings, "SAARAS_ENABLED", True)
    
    mock_transcribe.return_value = {
        "transcript": "   ",
        "language_code": "en-IN"
    }
    
    with pytest.raises(STTServiceError) as exc:
        stt_service.transcribe(b"dummy", "test.wav")
    assert exc.value.status_code == 400

@patch("backend.pipeline.stt.transcribe_audio")
def test_saaras_error_propagation(mock_transcribe, monkeypatch):
    monkeypatch.setattr(settings, "SAARAS_ENABLED", True)
    
    mock_transcribe.side_effect = SaarasAPIError("Upstream STT timeout", status_code=504)
    
    with pytest.raises(STTServiceError) as exc:
        stt_service.transcribe(b"dummy", "test.wav")
    assert exc.value.status_code == 504

def test_language_normalization():
    assert stt_service._normalize_language("hi-IN") == "hi"
    assert stt_service._normalize_language("en-US") == "en"
    assert stt_service._normalize_language("bn-IN") == "bn"
    assert stt_service._normalize_language("unknown") is None
    assert stt_service._normalize_language(None) is None
