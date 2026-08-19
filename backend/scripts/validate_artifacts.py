import os
import sys

# Ensure backend can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.artifact_loader import ArtifactLoader
from backend.config import settings

def main():
    print("=" * 40)
    print("HHG ARTIFACT VALIDATION")
    print("=" * 40)
    print()
    print(f"Artifact root:\n{settings.HHG_ARTIFACT_DIR}")
    print()
    
    loader = ArtifactLoader()
    loader.initialize()
    
    status = loader.status
    
    print("Manifest:")
    print("PASS" if status.get("manifest") else "FAIL")
    print()
    
    print("BM25:")
    print("PASS" if status.get("bm25") else "FAIL")
    print()
    
    print("HNSW:")
    print("PASS" if status.get("hnsw") else "FAIL")
    print()
    
    print("Metadata:")
    print("PASS" if status.get("metadata") else "FAIL")
    print()
    
    print("ONNX:")
    print("PASS" if status.get("onnx") else "FAIL")
    print()
    
    print("Validation summary:")
    print("PASS" if status.get("validation_summary") else "FAIL")
    print()
    
    print("Checksums:")
    c_status = status.get("checksums")
    if c_status is True:
        print("PASS")
    elif c_status == "not_provided":
        print("NOT PROVIDED")
    else:
        print("FAIL")
    print()
    
    dim = loader.manifest_data.get("embedding_dimension", 384) if status.get("manifest") else "N/A"
    print(f"Embedding dimension:\n{dim}")
    print()
    
    space = loader.manifest_data.get("hnsw_space", "cosine") if status.get("manifest") else "N/A"
    print(f"HNSW space:\n{space}")
    print()
    
    lang = loader.manifest_data.get("language", "hi") if status.get("manifest") else "N/A"
    print(f"Language:\n{lang}")
    print()
    
    print("=" * 40)
    final_status = "READY" if status.get("valid") else "NOT READY"
    print(f"Final status:\n{final_status}")
    
    if loader.errors:
        print("Errors:")
        for e in loader.errors:
            print(f"- {e}")
    print("=" * 40)

if __name__ == "__main__":
    main()
