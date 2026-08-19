import pytest
import responses
import requests
from backend.config import settings
from backend.pipeline.saaras_client import transcribe_audio, SaarasAPIError

@responses.activate
def test_transcribe_audio_success(monkeypatch):
    monkeypatch.setattr(settings, "SAARAS_ENABLED", True)
    monkeypatch.setattr(settings, "SAARAS_API_KEY", "dummy_key")
    monkeypatch.setattr(settings, "SAARAS_BASE_URL", "http://fake-stt.com/transcribe")
    
    responses.add(
        responses.POST,
        "http://fake-stt.com/transcribe",
        json={"transcript": "hello", "language_code": "en-IN"},
        status=200
    )
    
    result = transcribe_audio(b"audio data", "test.webm")
    assert result["transcript"] == "hello"
    
    # Verify Content-Type parsing logic
    assert "audio/webm" in responses.calls[0].request.body.decode('utf-8', errors='ignore')

@responses.activate
def test_transcribe_audio_http_failure(monkeypatch):
    monkeypatch.setattr(settings, "SAARAS_ENABLED", True)
    monkeypatch.setattr(settings, "SAARAS_API_KEY", "dummy_key")
    monkeypatch.setattr(settings, "SAARAS_BASE_URL", "http://fake-stt.com/transcribe")
    
    responses.add(
        responses.POST,
        "http://fake-stt.com/transcribe",
        body="Bad Gateway",
        status=502
    )
    
    with pytest.raises(SaarasAPIError) as exc:
        transcribe_audio(b"audio data", "test.wav")
    assert exc.value.status_code == 502
    assert "Upstream STT failed" in str(exc.value)

@responses.activate
def test_transcribe_audio_malformed_json(monkeypatch):
    monkeypatch.setattr(settings, "SAARAS_ENABLED", True)
    monkeypatch.setattr(settings, "SAARAS_API_KEY", "dummy_key")
    monkeypatch.setattr(settings, "SAARAS_BASE_URL", "http://fake-stt.com/transcribe")
    
    responses.add(
        responses.POST,
        "http://fake-stt.com/transcribe",
        body="{malformed json",
        status=200
    )
    
    with pytest.raises(SaarasAPIError) as exc:
        transcribe_audio(b"audio data", "test.wav")
    assert exc.value.status_code == 502
    assert "Invalid response" in str(exc.value)

@responses.activate
def test_transcribe_audio_timeout(monkeypatch):
    monkeypatch.setattr(settings, "SAARAS_ENABLED", True)
    monkeypatch.setattr(settings, "SAARAS_API_KEY", "dummy_key")
    monkeypatch.setattr(settings, "SAARAS_BASE_URL", "http://fake-stt.com/transcribe")
    
    responses.add(
        responses.POST,
        "http://fake-stt.com/transcribe",
        body=requests.exceptions.Timeout("Connection timed out")
    )
    
    with pytest.raises(SaarasAPIError) as exc:
        transcribe_audio(b"audio data", "test.wav")
    assert exc.value.status_code == 504
    assert "timeout" in str(exc.value).lower()
