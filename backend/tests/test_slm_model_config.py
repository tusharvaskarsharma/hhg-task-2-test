import pytest
import os
from backend.config import Settings

def test_slm_model_config_default():
    """
    Test that SLM model defaults to llama-3.1-8b-instant and 
    the provider is groq, enforcing the latency requirement.
    """
    # Temporarily remove env vars to test defaults
    old_model = os.environ.get("HHG_SLM_MODEL")
    old_provider = os.environ.get("HHG_SLM_PROVIDER")
    
    if "HHG_SLM_MODEL" in os.environ:
        del os.environ["HHG_SLM_MODEL"]
    if "HHG_SLM_PROVIDER" in os.environ:
        del os.environ["HHG_SLM_PROVIDER"]
        
    s = Settings()
    
    assert s.SLM_MODEL == "llama-3.1-8b-instant"
    assert s.SLM_PROVIDER == "groq"
    
    if old_model:
        os.environ["HHG_SLM_MODEL"] = old_model
    if old_provider:
        os.environ["HHG_SLM_PROVIDER"] = old_provider
