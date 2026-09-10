"""Embedding model config."""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """Wrapper around sentence-transformers for text embeddings."""

    def __init__(self) -> None:
        """Load the embedding model once for reuse across calls."""
        self._model = SentenceTransformer(MODEL_NAME)

    def embed_text(self, text: str) -> list[float]:
        """Return the embedding vector for a single text string.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        embedding = self._model.encode(text)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for multiple texts in one batch.

        Args:
            texts: List of input strings to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings = self._model.encode(texts)
        return embeddings.tolist()
