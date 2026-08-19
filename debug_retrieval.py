import sys
sys.path.append("e:\\HHG_T2\\HHG")

from backend.artifact_loader import loader_instance
import asyncio

def test():
    loader_instance.initialize()
    
    from backend.pipeline.retrieval_service import RetrievalService
    retriever = RetrievalService()
    retriever.initialize()
    
    for lang in ['hi', 'bn', 'en']:
        print(f"\n--- Testing {lang} ---")
        query = "hello world" if lang == 'en' else "नमस्ते"
        try:
            results = retriever.execute_query(query, lang, top_k=2)
            print(f"Results for {lang}: {len(results.get('results', []))}")
            if len(results.get('results', [])) == 0:
                df = loader_instance.get_metadata(lang)
                id_col = next((c for c in ["id", "doc_id", "passage_id"] if c in df.columns), df.columns[0])
                print(f"DataFrame for {lang} has {len(df)} rows. ID col: {id_col}")
                
                # Let's run just HNSW to see its IDs
                from backend.pipeline.dense_retriever import HNSWRetriever
                hnsw = HNSWRetriever()
                import numpy as np
                dummy_vector = np.random.rand(384).astype(np.float32)
                res = hnsw.retrieve(dummy_vector, lang, 2)
                
                hnsw_ids = [r['id'] for r in res]
                print(f"HNSW returned IDs: {hnsw_ids}")
                
                df_ids = df[id_col].astype(str).values
                are_in = [i in df_ids for i in hnsw_ids]
                print(f"Are these IDs in the DF? {are_in}")
                
                if not all(are_in):
                    print("Some IDs are NOT in the DF! Here is a sample of DF IDs:")
                    print(df_ids[:10])
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test()
