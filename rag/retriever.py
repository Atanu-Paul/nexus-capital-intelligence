"""Retrieval logic used by agents."""

from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore


class Retriever:
    """Retrieve relevant document chunks from the vector store."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        """Initialize the retriever with a vector store and embedding model."""
        self._vector_store = vector_store or VectorStore()
        self._embedding_model = embedding_model or EmbeddingModel()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        sector_filter: str | None = None,
    ) -> list[dict]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: Natural language search query.
            top_k: Maximum number of chunks to return.
            sector_filter: Optional sector name to filter results.

        Returns:
            Top matching chunks with metadata and similarity scores.
        """
        query_embedding = self._embedding_model.embed_text(query)
        return self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            sector_filter=sector_filter,
        )
