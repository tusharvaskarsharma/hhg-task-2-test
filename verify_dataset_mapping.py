import pandas as pd
from datasets import load_dataset
import re

def normalize(text):
    # Strip whitespace, lowercase, remove punctuation for robust matching
    t = str(text).lower().strip()
    return re.sub(r'[^\w\s]', '', t)

def verify_mapping():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("Loading MIRACL HI dataset...")
    ds = load_dataset('miracl/miracl', 'hi', split='dev', trust_remote_code=True)
    
    print("Loading Artifact Metadata...")
    df = pd.read_parquet('E:\\HHG_T2\\HHG\\hhg_rag_artifacts\\metadata\\passage_metadata.parquet', columns=['text'])
    
    # Pre-normalize all artifact texts for fast lookup
    print("Normalizing artifact texts...")
    artifact_texts_raw = df['text'].tolist()
    artifact_texts = set([normalize(t) for t in artifact_texts_raw])
    
    print(f"Artifact corpus size: {len(artifact_texts_raw)}")
    
    matches = 0
    total = 0
    false_negatives = []
    
    print("Evaluating Top 100 Queries mapping...")
    for i in range(100):
        row = ds[i]
        pos_passages = row.get("positive_passages", [])
        if not pos_passages:
            continue
            
        for p in pos_passages:
            total += 1
            gold = normalize(p['text'])
            
            if gold in artifact_texts:
                matches += 1
            else:
                # Check for partial matches or overlap
                partial_match = False
                for at in artifact_texts:
                    if gold in at or at in gold:
                        partial_match = True
                        matches += 1
                        break
                
                if not partial_match:
                    false_negatives.append({
                        "query_id": row["query_id"],
                        "docid": p["docid"],
                        "text": p["text"]
                    })
                    
    print(f"\n--- MAPPING RESULTS ---")
    print(f"Total Evaluated Positive Passages: {total}")
    print(f"Exact or Substring Matches Found: {matches}")
    print(f"False Negatives (Unmapped): {len(false_negatives)}")
    if total > 0:
        print(f"Mapping Reliability Rate: {matches/total * 100:.2f}%")
    
    if false_negatives:
        print("\nFirst 3 Unmapped Examples:")
        for fn in false_negatives[:3]:
            print(f"Query ID: {fn['query_id']} | DocID: {fn['docid']}")

if __name__ == "__main__":
    verify_mapping()
