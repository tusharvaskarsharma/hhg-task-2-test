import json
from pathlib import Path
from typing import Dict, List, Set, Optional

import pandas as pd

class GroundTruthError(RuntimeError):
    pass

class GroundTruth:
    def __init__(self):
        self.queries: List[dict] = []
        self.by_language: Dict[str, List[dict]] = {"hi": [], "en": [], "bn": []}
        self.supported_queries_by_language: Dict[str, List[dict]] = {"hi": [], "en": [], "bn": []}
        self.unsupported_queries_by_language: Dict[str, List[dict]] = {"hi": [], "en": [], "bn": []}
        self.relevant_ids_by_query: Dict[str, Set[str]] = {}

def load_ground_truth(path: Path, artifact_root: Optional[Path] = None) -> GroundTruth:
    if not path.is_file():
        raise GroundTruthError(f"Ground truth file not found: {path}")
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise GroundTruthError(f"Failed to parse ground truth JSON: {e}")
        
    schema_version = data.get("schema_version")
    if schema_version != "hhg-ground-truth-upload-v1":
        raise GroundTruthError(f"Invalid schema version: {schema_version}")
        
    source = data.get("source", {})
    if source.get("dataset") != "ai4bharat/MSMARCO-XI":
        raise GroundTruthError("Dataset must be ai4bharat/MSMARCO-XI")
        
    gt = GroundTruth()
    seen_query_ids = {"hi": set(), "en": set(), "bn": set()}
    
    queries = data.get("queries", [])
    for q in queries:
        q_id = q.get("query_id")
        lang = q.get("language")
        query_text = q.get("query")
        gold_answer = q.get("gold_answer")
        relevant_ids = q.get("relevant_passage_ids")
        
        if lang not in ["hi", "en", "bn"]:
            raise GroundTruthError(f"Unsupported language: {lang}")
            
        if not q_id or not query_text or gold_answer is None or relevant_ids is None:
            raise GroundTruthError(f"Missing required query fields for {q_id}")
            
        if q_id in seen_query_ids[lang]:
            raise GroundTruthError(f"Duplicate query ID: {q_id} in {lang}")
            
        seen_query_ids[lang].add(q_id)
        
        q["relevant_passage_ids"] = [str(i) for i in relevant_ids]
        
        gt.queries.append(q)
        gt.by_language[lang].append(q)
        gt.relevant_ids_by_query[q_id] = set(q["relevant_passage_ids"])
        
        if len(q["relevant_passage_ids"]) > 0:
            gt.supported_queries_by_language[lang].append(q)
        else:
            gt.unsupported_queries_by_language[lang].append(q)
            
    if artifact_root:
        _validate_ids_with_artifacts(gt, artifact_root)
        
    return gt

def _validate_ids_with_artifacts(gt: GroundTruth, artifact_root: Path):
    for lang in ["hi", "en", "bn"]:
        if not gt.supported_queries_by_language[lang]:
            continue
            
        # Find parquet file
        lang_root = artifact_root / lang if (artifact_root / lang).is_dir() else artifact_root
        
        candidates = [
            lang_root / "metadata" / "passage_metadata.parquet",
            lang_root / "metadata" / "data.parquet",
            lang_root / "metadata.parquet",
        ]
        
        metadata_path = None
        for p in candidates:
            if p.is_file():
                metadata_path = p
                break
                
        if not metadata_path:
            metadata_dir = lang_root / "metadata"
            if metadata_dir.is_dir():
                parquets = list(metadata_dir.glob("*.parquet"))
                if parquets:
                    metadata_path = parquets[0]
                    
        if not metadata_path:
            raise GroundTruthError(f"Could not find metadata for {lang}")
            
        df = pd.read_parquet(metadata_path)
        if "language" in df.columns:
            df = df[df["language"] == lang]
            
        id_col = "passage_id" if "passage_id" in df.columns else "id"
        if id_col not in df.columns:
            id_col = [c for c in df.columns if c in ("doc_id", "chunk_id")][0]
            
        valid_ids = set(df[id_col].astype(str))
        
        for q in gt.supported_queries_by_language[lang]:
            if not q["relevant_passage_ids"]:
                raise GroundTruthError(f"Empty relevant IDs for supported query {q['query_id']}")
            for rel_id in q["relevant_passage_ids"]:
                if rel_id not in valid_ids:
                    raise GroundTruthError(f"ID {rel_id} for query {q['query_id']} not found in {lang} metadata")
