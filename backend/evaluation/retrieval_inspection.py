from typing import List, Dict, Any

def analyze_failures(query_evals: List[Dict[str, Any]]) -> Dict[str, Any]:
    bm25_only_wins = 0
    hnsw_only_wins = 0
    rrf_wins = 0 # RRF hits when both missed, or RRF hits at higher rank? We'll just define 'wins' as hitting at K=10 for simplicity in this analysis, or we can use RR. Let's use hit@10.
    all_miss = 0
    
    # Let's use hit_at_10 as the baseline for "win" / "miss" for simplicity
    for q in query_evals:
        bm25_hit = q["bm25"]["hit_at_10"]
        hnsw_hit = q["hnsw"]["hit_at_10"]
        rrf_hit = q["rrf"]["hit_at_10"]
        
        if not bm25_hit and not hnsw_hit and not rrf_hit:
            all_miss += 1
        elif bm25_hit and not hnsw_hit:
            bm25_only_wins += 1
        elif hnsw_hit and not bm25_hit:
            hnsw_only_wins += 1
        elif rrf_hit and not bm25_hit and not hnsw_hit:
            rrf_wins += 1 # RRF pulled it up!
            
    return {
        "bm25_only_wins": bm25_only_wins,
        "hnsw_only_wins": hnsw_only_wins,
        "rrf_wins": rrf_wins,
        "all_miss": all_miss
    }

def print_failure_analysis(analysis: Dict[str, Any]):
    print("=" * 40)
    print("RETRIEVAL FAILURE ANALYSIS")
    print("=" * 40)
    print()
    print(f"BM25 only wins:\n{analysis['bm25_only_wins']}\n")
    print(f"HNSW only wins:\n{analysis['hnsw_only_wins']}\n")
    print(f"RRF wins (both individuals missed):\n{analysis['rrf_wins']}\n")
    print(f"All systems miss:\n{analysis['all_miss']}\n")
    print("=" * 40)

def inspect_query(q_eval: Dict[str, Any]):
    print(f"Query:\n{q_eval['query']}")
    print()
    print(f"Gold IDs:\n{q_eval['gold_ids']}")
    print()
    
    def print_top(system_name, ids):
        print(f"{system_name}:")
        for i, doc_id in enumerate(ids[:5]): # Print top 5
            mark = " [RELEVANT]" if doc_id in q_eval["gold_ids"] else ""
            print(f"{i+1}. ID {doc_id}{mark}")
        print()
        
    print_top("BM25", q_eval["bm25"]["retrieved_ids"])
    print_top("HNSW", q_eval["hnsw"]["retrieved_ids"])
    print_top("RRF", q_eval["rrf"]["retrieved_ids"])

def sample_inspection(query_evals: List[Dict[str, Any]]):
    # Try to find specific cases
    samples = []
    
    # 1. successful RRF cases (where RRF hits but maybe one missed, or RRF rank is 1)
    rrf_success = [q for q in query_evals if q["rrf"]["hit_at_10"]]
    if rrf_success: samples.append(("Successful RRF", rrf_success[0]))
    
    # 2. BM25-only successes
    bm25_only = [q for q in query_evals if q["bm25"]["hit_at_10"] and not q["hnsw"]["hit_at_10"]]
    if bm25_only: samples.append(("BM25-only Success", bm25_only[0]))
    
    # 3. HNSW-only successes
    hnsw_only = [q for q in query_evals if q["hnsw"]["hit_at_10"] and not q["bm25"]["hit_at_10"]]
    if hnsw_only: samples.append(("HNSW-only Success", hnsw_only[0]))
    
    # 4. RRF failures (BM25 or HNSW hit, but RRF missed)
    rrf_fail = [q for q in query_evals if not q["rrf"]["hit_at_10"] and (q["bm25"]["hit_at_10"] or q["hnsw"]["hit_at_10"])]
    if rrf_fail: samples.append(("RRF Failure (Pulled down)", rrf_fail[0]))
    
    # 5. Complete failures
    all_fail = [q for q in query_evals if not q["bm25"]["hit_at_10"] and not q["hnsw"]["hit_at_10"] and not q["rrf"]["hit_at_10"]]
    if all_fail: samples.append(("Complete Failure", all_fail[0]))
    
    # Fill remaining to get to 10
    existing_ids = {q["query_id"] for _, q in samples}
    for q in query_evals:
        if len(samples) >= 10:
            break
        if q["query_id"] not in existing_ids:
            samples.append(("Random Sample", q))
            existing_ids.add(q["query_id"])
            
    print("=" * 40)
    print(f"SAMPLE INSPECTION ({len(samples)} queries)")
    print("=" * 40)
    print()
    
    for reason, q_eval in samples:
        print(f"--- Reason: {reason} ---")
        inspect_query(q_eval)
        print("-" * 40)
        print()
