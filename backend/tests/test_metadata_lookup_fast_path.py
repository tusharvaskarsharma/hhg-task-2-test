import pytest
from backend.pipeline.retrieval_service import RetrievalService
from backend.artifact_loader import loader_instance
import time

def test_metadata_lookup_fast_path():
    """
    Ensure the metadata maps are pre-built dictionaries (O(1) lookup)
    and not doing dynamic df.loc pandas lookups.
    """
    if not loader_instance.status.get("valid"):
        loader_instance.initialize()
        
    service = RetrievalService()
    service.initialize()
    
    assert hasattr(service, "metadata_maps")
    assert isinstance(service.metadata_maps, dict)
    
    # Must have populated at least one language map if valid
    if loader_instance.status.get("valid"):
        lang = loader_instance.SUPPORTED_LANGUAGES[0]
        assert lang in service.metadata_maps
        assert isinstance(service.metadata_maps[lang], dict)
        
        # Look up should be extremely fast
        if len(service.metadata_maps[lang]) > 0:
            sample_id = next(iter(service.metadata_maps[lang]))
            
            t0 = time.perf_counter()
            entry = service.metadata_maps[lang][sample_id]
            lookup_ms = (time.perf_counter() - t0) * 1000.0
            
            assert "text" in entry
            # Should be sub-millisecond
            assert lookup_ms < 1.0
