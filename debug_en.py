import sys
import os

from backend.artifact_loader import loader_instance
loader_instance.initialize()

from backend.pipeline.retrieval_service import retrieval_service
retrieval_service.initialize()

df = loader_instance.get_metadata('en')
print(f'en metadata size: {len(df)}')
print(f'en first few indices: {list(df.index[:5])}')

q_emb, _, _ = retrieval_service.embedder.embed_query('who is the pm of india')
hnsw_res = retrieval_service.hnsw.retrieve(q_emb, 'en', 5)

print(f'hnsw returned: {[r["id"] for r in hnsw_res]}')
