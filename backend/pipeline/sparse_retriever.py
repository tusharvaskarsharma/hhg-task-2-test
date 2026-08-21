import numpy as np
import json
import logging
import unicodedata
from pathlib import Path

from backend.artifact_loader import loader_instance

logger = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self):
        self._vocab_cache = {}

    @staticmethod
    def _unicode_words(text: str):
        """Tokenize like the uploaded ICU word-boundary artifact tokenizer.

        Python's default ``\\w`` splits Indic grapheme sequences incorrectly
        (for example, a Hindi word can be split around combining marks). The
        artifact tokenizer keeps letters, numbers, and combining marks together.
        """
        text = unicodedata.normalize("NFC", str(text)).casefold()
        words = []
        current = []

        def flush():
            if not current:
                return
            token = "".join(current)
            if (
                len(token) >= 2
                and any(unicodedata.category(c).startswith(("L", "N")) for c in token)
                and not token.isdigit()
            ):
                words.append(token)
            current.clear()

        for char in text:
            category = unicodedata.category(char)
            if category.startswith(("L", "N", "M")):
                current.append(char)
            else:
                flush()
        flush()
        return words

    def _vocabulary(self, language: str, bm25):
        vocab = getattr(bm25, "vocab_dict", None)
        if isinstance(vocab, dict) and vocab:
            return vocab
        if language not in self._vocab_cache:
            root = Path(loader_instance._artifact_root())
            path = root / "bm25" / language / "vocab.tokenizer.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            self._vocab_cache[language] = data.get("word_to_id", {})
        return self._vocab_cache[language]

    def _tokenize_with_artifact_vocab(self, query: str, language: str, bm25):
        vocab = self._vocabulary(language, bm25)
        tokens = []
        for token in self._unicode_words(query):
            if token in vocab:
                tokens.append(token)
            elif token.casefold() in vocab:
                tokens.append(token.casefold())
        return tokens

    def retrieve(self, query: str, language: str, top_k: int):
        bm25 = loader_instance.get_bm25(language)
        metadata = loader_instance.get_metadata(language)
        if bm25 is None:
            raise ValueError(f"BM25 index not initialized for {language}")
            
        try:
            # bm25s must receive tokens from the persisted corpus vocabulary.
            # Tokenizing each query with bm25s.tokenize() creates a new local
            # vocabulary, which corrupts token IDs and is especially harmful for
            # Hindi/Bengali. The exact artifact vocabulary is used here instead.
            if hasattr(bm25, "retrieve"):
                tokens = self._tokenize_with_artifact_vocab(query, language, bm25)
                if not tokens:
                    return []
                results_idx, scores = bm25.retrieve(
                    [tokens],
                    corpus=None,
                    k=top_k,
                    show_progress=False,
                    leave_progress=False,
                )
                top_k_indices = results_idx[0]
                scores = scores[0]
            else:
                tokenized_query = self._unicode_words(query)
                scores = bm25.get_scores(tokenized_query)
                top_k_indices = np.argsort(scores)[::-1][:top_k]
                scores = scores[top_k_indices]
        except Exception as e:
            raise ValueError(f"Failed to get scores from BM25 object: {e}")
            
        results = []
        for rank, (idx, score) in enumerate(zip(top_k_indices, scores)):
            if score <= 0:
                continue
            doc_id = str(metadata.index[idx])
            results.append({
                "id": doc_id,
                "score": float(score),
                "rank": rank + 1,
                "source": "bm25"
            })
            
        return results
