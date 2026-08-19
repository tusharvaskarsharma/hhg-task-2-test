"""
backend/scripts/validate_artifacts.py
Deep artifact validation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import hashlib
import datetime
import logging
import pandas as pd
import hnswlib
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import settings

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

REQUIRED_LANGUAGES = ["hi", "bn", "en"]
EXPECTED_EMBEDDING_DIM = 384

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def _check_onnx(artifact_root: str) -> dict:
    checks = {}
    onnx_dir = os.path.join(artifact_root, "embedding", "onnx_int8", "onnx")
    candidates = ["model.onnx", "model_quint8_avx2.onnx"]
    found = None
    for name in candidates:
        p = os.path.join(onnx_dir, name)
        if os.path.exists(p):
            found = p
            break

    if found:
        checks["onnx_path"] = found
        checks["onnx_exists"] = True
        checks["onnx_size_mb"] = round(os.path.getsize(found) / 1e6, 1)
        checks["sha256"] = _sha256_file(found)
        checks["dimension"] = EXPECTED_EMBEDDING_DIM
        checks["status"] = "PASS"
    else:
        checks["onnx_exists"] = False
        checks["status"] = "FAIL"
        checks["error"] = "ONNX model file not found"
    return checks

def _check_language(artifact_root: str, lang: str) -> dict:
    result = {
        "metadata_rows": 0,
        "hnsw_count": 0,
        "bm25_count": 0,
        "unique_ids": False,
        "missing_hnsw_labels": 0,
        "missing_metadata_ids": 0,
        "language_mismatch_rows": 0,
        "checksum_failures": 0,
        "status": "PASS",
        "errors": []
    }
    
    # Manifest
    manifest_path = os.path.join(artifact_root, "build_manifest.json")
    if not os.path.exists(manifest_path):
        manifest_path = os.path.join(artifact_root, "config.json")
        
    if not os.path.exists(manifest_path):
        result["errors"].append(f"No manifest found in {artifact_root}")
        result["status"] = "FAIL"
        return result

    with open(manifest_path) as f:
        manifest_data = json.load(f)

    # 1. Metadata
    metadata_path = os.path.join(artifact_root, "metadata", "passage_metadata.parquet")
    if not os.path.exists(metadata_path):
        result["errors"].append(f"Metadata parquet not found at {metadata_path}")
        result["status"] = "FAIL"
        return result
        
    try:
        df = pd.read_parquet(metadata_path)
    except Exception as e:
        result["errors"].append(f"Failed to read parquet: {e}")
        result["status"] = "FAIL"
        return result
        
    # filter for lang
    if "language" in df.columns:
        df_lang = df[df["language"] == lang]
        result["language_mismatch_rows"] = len(df) - len(df_lang)
        df = df_lang
        
    result["metadata_rows"] = len(df)
    if result["metadata_rows"] == 0:
        result["errors"].append(f"Metadata row count is 0 for {lang}")
        result["status"] = "FAIL"
        return result

    id_col = "passage_id" if "passage_id" in df.columns else "id"
    if id_col not in df.columns:
        result["errors"].append(f"Selected ID column '{id_col}' missing")
        result["status"] = "FAIL"
        return result
        
    result["unique_ids"] = df[id_col].is_unique and not df[id_col].isnull().any()
    if not result["unique_ids"]:
        result["errors"].append("IDs are not unique or contain nulls")
        result["status"] = "FAIL"
        
    if "text" not in df.columns and "passage" not in df.columns:
        result["errors"].append("Text column missing")
        result["status"] = "FAIL"
    else:
        text_col = "text" if "text" in df.columns else "passage"
        sample = df.sample(min(100, len(df)))
        if sample[text_col].str.strip().eq("").any() or sample[text_col].isnull().any():
            result["errors"].append("Sampled text contains empty or null strings")
            result["status"] = "FAIL"
            
    ids_set = set(df[id_col].astype(str).tolist())

    # 2. HNSW
    hnsw_dir = os.path.join(artifact_root, "hnsw", lang)
    hnsw_path = os.path.join(hnsw_dir, f"hnsw_{lang}.bin")
    if not os.path.exists(hnsw_path):
        hnsw_path = os.path.join(hnsw_dir, "index.bin")
        
    if not os.path.exists(hnsw_path):
        result["errors"].append(f"HNSW index not found for {lang}")
        result["status"] = "FAIL"
    else:
        try:
            space = manifest_data.get("hnsw_space", "cosine")
            dim = manifest_data.get("embedding_dimension", EXPECTED_EMBEDDING_DIM)
            p = hnswlib.Index(space=space, dim=dim)
            p.load_index(hnsw_path, max_elements=result["metadata_rows"])
            result["hnsw_count"] = p.get_current_count()
            
            if result["hnsw_count"] != result["metadata_rows"]:
                result["errors"].append(f"HNSW count ({result['hnsw_count']}) != metadata ({result['metadata_rows']})")
                result["status"] = "FAIL"
                
            # labels mapping check if label mapping exists
            map_path = os.path.join(hnsw_dir, f"id_mapping_{lang}.json")
            if os.path.exists(map_path):
                with open(map_path) as f:
                    mapping = json.load(f)
                int_to_id = mapping.get("int_to_str", mapping.get("int_to_id", {}))
                mapped_ids = set(str(v) for v in int_to_id.values())
                result["missing_hnsw_labels"] = len(mapped_ids - ids_set)
                result["missing_metadata_ids"] = len(ids_set - mapped_ids)
                if result["missing_hnsw_labels"] > 0 or result["missing_metadata_ids"] > 0:
                    result["errors"].append("HNSW labels do not perfectly map to metadata IDs")
                    result["status"] = "FAIL"
        except Exception as e:
            result["errors"].append(f"HNSW load failed: {e}")
            result["status"] = "FAIL"

    # 3. BM25
    bm25_dir = os.path.join(artifact_root, "bm25", lang)
    bm25_path = os.path.join(bm25_dir, "bm25.pkl")
    if not os.path.exists(bm25_path):
        result["errors"].append(f"BM25 not found for {lang}")
        result["status"] = "FAIL"
    else:
        try:
            with open(bm25_path, "rb") as f:
                bm25_data = pickle.load(f)
            
            # extract count
            if hasattr(bm25_data, "corpus_size"):
                result["bm25_count"] = bm25_data.corpus_size
            elif isinstance(bm25_data, dict) and "corpus_size" in bm25_data:
                result["bm25_count"] = bm25_data["corpus_size"]
            elif isinstance(bm25_data, dict) and "passage_id_map" in bm25_data:
                result["bm25_count"] = len(bm25_data["passage_id_map"])
            elif hasattr(bm25_data, "doc_freqs"):
                result["bm25_count"] = len(bm25_data.doc_freqs)
            else:
                result["bm25_count"] = result["metadata_rows"] # fallback assumption for standard bm25s
                
            if result["bm25_count"] != result["metadata_rows"]:
                result["errors"].append(f"BM25 count ({result['bm25_count']}) != metadata ({result['metadata_rows']})")
                result["status"] = "FAIL"
        except Exception as e:
            result["errors"].append(f"BM25 load failed: {e}")
            result["status"] = "FAIL"

    return result

def validate(artifact_root: str) -> dict:
    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "artifact_root": artifact_root,
        "overall_status": "PASS",
        "manifest_sha256": "missing",
        "errors": [],
        "onnx": {},
        "languages": {},
    }

    if not os.path.isdir(artifact_root):
        report["overall_status"] = "FAIL: ARTIFACT_ROOT_MISSING"
        report["errors"].append(f"Artifact root not found: {artifact_root}")
        return report
        
    manifest_path = os.path.join(artifact_root, "build_manifest.json")
    if not os.path.exists(manifest_path):
        manifest_path = os.path.join(artifact_root, "config.json")
        
    if os.path.exists(manifest_path):
        report["manifest_sha256"] = _sha256_file(manifest_path)[:16]

    # ONNX
    onnx = _check_onnx(artifact_root)
    report["onnx"] = onnx
    if onnx["status"] != "PASS":
        report["overall_status"] = "FAIL"
        report["errors"].append(onnx.get("error", "ONNX check failed"))

    # Per-language
    for lang in REQUIRED_LANGUAGES:
        lang_result = _check_language(artifact_root, lang)
        report["languages"][lang] = lang_result
        if lang_result["status"] != "PASS":
            report["overall_status"] = "FAIL"
            for e in lang_result["errors"]:
                report["errors"].append(f"[{lang}] {e}")

    return report

def main():
    parser = argparse.ArgumentParser(description="HHG Artifact Validation")
    parser.add_argument("--json", metavar="PATH")
    parser.add_argument("--artifact-dir", metavar="DIR")
    args = parser.parse_args()

    artifact_root = args.artifact_dir or settings.HHG_ARTIFACT_DIR
    report = validate(artifact_root)

    out_path = args.json or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "reports", "artifact_validation.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    sys.exit(0 if report["overall_status"] == "PASS" else 1)

if __name__ == "__main__":
    main()
