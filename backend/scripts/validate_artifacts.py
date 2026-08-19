"""
backend/scripts/validate_artifacts.py
Phase 12 — Machine-readable artifact validation.
Outputs a JSON report and exits nonzero on any failed invariant.

Usage:
    python backend/scripts/validate_artifacts.py
    python backend/scripts/validate_artifacts.py --json reports/artifact_validation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import hashlib
import datetime
import logging

# Allow running directly: python backend/scripts/validate_artifacts.py
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
    # Accept either model name
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
        checks["onnx_sha256"] = _sha256_file(found)[:16]
        checks["status"] = "PASS"
    else:
        checks["onnx_path"] = onnx_dir
        checks["onnx_exists"] = False
        checks["status"] = "FAIL"
        checks["error"] = "ONNX model file not found"
    return checks


def _check_language(artifact_root: str, lang: str) -> dict:
    result = {"language": lang, "checks": {}, "status": "PASS", "errors": []}
    checks = result["checks"]

    # BM25
    bm25_dir = os.path.join(artifact_root, "bm25", lang)
    checks["bm25_dir_exists"] = os.path.isdir(bm25_dir)
    if checks["bm25_dir_exists"]:
        bm25_files = os.listdir(bm25_dir)
        checks["bm25_files"] = bm25_files
        checks["bm25_nonempty"] = len(bm25_files) > 0
        if not checks["bm25_nonempty"]:
            result["errors"].append(f"BM25 dir {bm25_dir} is empty")
    else:
        result["errors"].append(f"BM25 dir missing: {bm25_dir}")

    # HNSW
    hnsw_dir = os.path.join(artifact_root, "hnsw", lang)
    hnsw_candidates = ["index.bin", f"hnsw_{lang}.bin"]
    hnsw_found = None
    if os.path.isdir(hnsw_dir):
        for name in hnsw_candidates:
            p = os.path.join(hnsw_dir, name)
            if os.path.exists(p):
                hnsw_found = p
                break
        if not hnsw_found:
            # check for any .bin file
            for f in os.listdir(hnsw_dir):
                if f.endswith(".bin"):
                    hnsw_found = os.path.join(hnsw_dir, f)
                    break
    checks["hnsw_exists"] = hnsw_found is not None
    if hnsw_found:
        checks["hnsw_path"] = hnsw_found
        checks["hnsw_size_mb"] = round(os.path.getsize(hnsw_found) / 1e6, 1)
    else:
        result["errors"].append(f"HNSW index not found in {hnsw_dir}")

    # Manifest
    # Check both language-specific and root manifest
    manifest_candidates = [
        os.path.join(artifact_root, lang, "manifest.json"),
        os.path.join(artifact_root, "build_manifest.json"),
        os.path.join(artifact_root, "config.json"),
    ]
    manifest_found = None
    for mp in manifest_candidates:
        if os.path.exists(mp):
            manifest_found = mp
            break
    checks["manifest_found"] = manifest_found is not None
    if manifest_found:
        checks["manifest_path"] = manifest_found
        with open(manifest_found) as f:
            manifest_data = json.load(f)
        checks["manifest_keys"] = list(manifest_data.keys())
    else:
        result["errors"].append(f"No manifest found for language {lang}")

    if result["errors"]:
        result["status"] = "FAIL"

    return result


def validate(artifact_root: str) -> dict:
    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "artifact_root": artifact_root,
        "overall_status": "PASS",
        "errors": [],
        "onnx": {},
        "languages": {},
    }

    if not os.path.isdir(artifact_root):
        report["overall_status"] = "FAIL"
        report["errors"].append(f"Artifact root not found: {artifact_root}")
        return report

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
    parser.add_argument(
        "--json", metavar="PATH",
        help="Write JSON report to this path (default: reports/artifact_validation.json)"
    )
    parser.add_argument(
        "--artifact-dir", metavar="DIR",
        help="Override artifact root directory"
    )
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

    # Print summary
    print(json.dumps(report, indent=2))
    print(f"\nReport written to: {out_path}")
    print(f"Overall status: {report['overall_status']}")

    if report["errors"]:
        print("Errors:")
        for e in report["errors"]:
            print(f"  - {e}")

    sys.exit(0 if report["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
