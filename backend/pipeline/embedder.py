import numpy as np
import logging
from backend.artifact_loader import loader_instance
import warnings
warnings.filterwarnings("ignore")

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self):
        self.session = loader_instance.onnx_session
        if AutoTokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
        else:
            self.tokenizer = None

    def embed_query(self, query: str) -> tuple[np.ndarray, float, float]:
        import time
        if not self.session:
            raise ValueError("ONNX session not initialized.")
            
        if self.tokenizer is None:
            raise RuntimeError("Transformers library is not available for tokenization.")
            
        formatted_query = f"query: {query}"
        
        t0 = time.perf_counter()
        inputs = self.tokenizer(
            formatted_query, 
            return_tensors="np", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        tokenization_ms = (time.perf_counter() - t0) * 1000.0
        
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        
        input_names = [i.name for i in self.session.get_inputs()]
        if "token_type_ids" in input_names:
            if "token_type_ids" in inputs:
                onnx_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
            else:
                onnx_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"]).astype(np.int64)
            
        t1 = time.perf_counter()
        outputs = self.session.run(None, onnx_inputs)
        
        last_hidden_state = outputs[0]
        attention_mask = inputs["attention_mask"]
        
        input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), last_hidden_state.shape)
        embeddings = np.sum(last_hidden_state * input_mask_expanded, 1) / np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=None)
        
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        embedding_ms = (time.perf_counter() - t1) * 1000.0
        
        return embeddings[0].astype(np.float32), tokenization_ms, embedding_ms
