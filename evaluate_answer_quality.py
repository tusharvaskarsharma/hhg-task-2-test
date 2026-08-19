import sys
import pandas as pd
from backend.pipeline.generator import generator_service
from backend.pipeline.sparse_retriever import BM25Retriever
from backend.pipeline.dense_retriever import HNSWRetriever
from backend.pipeline.fusion import rrf_fuse
from backend.pipeline.embedder import Embedder

def evaluate_answers():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    from dotenv import load_dotenv
    load_dotenv()
    from backend.artifact_loader import loader_instance
    loader_instance.initialize()
    
    embedder = Embedder()
    bm25 = BM25Retriever()
    hnsw = HNSWRetriever()
    
    # 5 Hand-picked factual queries in Hindi and English with expected short answers
    eval_set = [
        {
            "query": "भारत की राजधानी क्या है?",
            "lang": "hi",
            "expected_snippet": "नई दिल्ली",
            "type": "factual"
        },
        {
            "query": "Who discovered gravity?",
            "lang": "en",
            "expected_snippet": "newton",
            "type": "factual"
        },
        {
            "query": "यह एक पूरी तरह से यादृच्छिक प्रश्न है जिसका उत्तर नहीं दिया जा सकता",
            "lang": "hi",
            "expected_snippet": "insufficient",
            "type": "refusal"
        }
    ]
    
    print("Evaluating Answer Quality...\n")
    
    results = {"correct": 0, "incorrect": 0, "refused": 0, "hallucinated": 0}
    
    for q in eval_set:
        print(f"Query: {q['query']}")
        q_vec, _, _ = embedder.embed_query(q['query'])
        b_res = bm25.retrieve(q['query'], q['lang'], top_k=5)
        h_res = hnsw.retrieve(q_vec, q['lang'], top_k=5)
        f_res = rrf_fuse(b_res, h_res, top_k=5)
        
        gen = generator_service.generate(q['query'], q['lang'], f_res)
        answer = (gen['answer'] or "").lower()
        state = gen['grounding'].get('status', 'UNSUPPORTED')
        
        print(f"Generated: {gen['answer']}")
        print(f"Grounding State: {state}")
        
        if q['type'] == "factual":
            if q['expected_snippet'].lower() in answer:
                if state == "SUPPORTED":
                    results['correct'] += 1
                    print("-> CORRECT (Supported)\n")
                else:
                    results['incorrect'] += 1
                    print("-> INCORRECT (State Mismatch)\n")
            else:
                if state == "INSUFFICIENT_CONTEXT":
                    results['refused'] += 1
                    print("-> REFUSED (Not in DB)\n")
                elif state == "UNSUPPORTED":
                    results['hallucinated'] += 1
                    print("-> HALLUCINATED (Generated without support)\n")
                else:
                    results['incorrect'] += 1
                    print("-> INCORRECT (Wrong Answer)\n")
        elif q['type'] == "refusal":
            if state == "INSUFFICIENT_CONTEXT":
                results['refused'] += 1
                print("-> CORRECT REFUSAL\n")
            else:
                results['incorrect'] += 1
                print("-> FAILED REFUSAL\n")
                
    print("\n--- ANSWER QUALITY RESULTS ---")
    for k, v in results.items():
        print(f"{k.upper()}: {v}")
        
if __name__ == "__main__":
    evaluate_answers()
