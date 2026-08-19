import pytest
from unittest.mock import patch, MagicMock
from backend.pipeline.slm_client import slm_client
from backend.config import settings

def test_slm_client_sends_correct_payload():
    original_enabled = slm_client.enabled
    original_model = slm_client.model
    slm_client.enabled = True
    slm_client.model = "llama-3.1-8b-instant"

    with patch.object(slm_client.session, 'post') as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "mocked answer"}}]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        answer = slm_client.generate("hello world")

        assert answer == "mocked answer"
        mock_post.assert_called_once()

        args, kwargs = mock_post.call_args
        payload = kwargs.get("json")

        assert payload is not None
        assert payload["model"] == "llama-3.1-8b-instant"
        assert payload["messages"] == [{"role": "user", "content": "hello world"}]
        assert payload["max_tokens"] == settings.SLM_MAX_TOKENS
        assert payload["temperature"] == settings.SLM_TEMPERATURE

    slm_client.enabled = original_enabled
    slm_client.model = original_model

def test_slm_client_disabled_no_outbound_calls():
    original_enabled = slm_client.enabled
    slm_client.enabled = False

    with patch.object(slm_client.session, 'post') as mock_post:
        from backend.pipeline.slm_client import SLMClientError
        with pytest.raises(SLMClientError) as exc_info:
            slm_client.generate("hello world")

        assert "SLM is disabled" in str(exc_info.value)
        mock_post.assert_not_called()

    slm_client.enabled = original_enabled
