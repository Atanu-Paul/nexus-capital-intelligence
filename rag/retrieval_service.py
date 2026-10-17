"""Thread-safe singleton retrieval service for concurrent agent access."""

import threading

from rag.embeddings import get_embedding_model
from rag.retriever import Retriever
from rag.vector_store import VectorStore

_lock = threading.Lock()
_vector_store: VectorStore | None = None
_retriever: Retriever | None = None
_cache: dict[tuple[str, str | None, int], list[dict]] = {}


def _init_retriever_if_needed() -> Retriever:
    """Initialize the shared retriever stack. Caller must hold ``_lock``."""
    global _vector_store, _retriever

    if _retriever is not None:
        return _retriever

    embedding_model = get_embedding_model()
    _vector_store = VectorStore(embedding_model=embedding_model)
    _retriever = Retriever(
        vector_store=_vector_store,
        embedding_model=embedding_model,
    )
    return _retriever


def retrieve_documents(
    query: str,
    top_k: int = 10,
    sector_filter: str | None = None,
    use_cache: bool = True,
) -> list[dict]:
    """Retrieve document chunks using a thread-safe, cached singleton stack."""
    cache_key = (query, sector_filter, top_k)

    with _lock:
        if use_cache:
            cached = _cache.get(cache_key)
            if cached is not None:
                return cached

        retriever = _init_retriever_if_needed()
        chunks = retriever.retrieve(
            query=query,
            top_k=top_k,
            sector_filter=sector_filter,
        )

        if use_cache:
            _cache[cache_key] = chunks

        return chunks
