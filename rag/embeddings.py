"""Embedding model config."""

import functools

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """Wrapper around sentence-transformers for text embeddings."""

    def __init__(self) -> None:
        """Load the embedding model once for reuse across calls."""
        self._model = SentenceTransformer(MODEL_NAME)

    def embed_text(self, text: str) -> list[float]:
        """Return the embedding vector for a single text string."""
        embedding = self._model.encode(text)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for multiple texts in one batch."""
        embeddings = self._model.encode(texts)
        return embeddings.tolist()


@functools.lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Return the process-wide singleton embedding model."""
    return EmbeddingModel()
