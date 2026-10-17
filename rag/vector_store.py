"""ChromaDB setup, add/query functions."""

import hashlib
import logging
from pathlib import Path

import chromadb

from rag.embeddings import EmbeddingModel, get_embedding_model
from rag.paths import chroma_db_dir

logger = logging.getLogger(__name__)

COLLECTION_NAME = "investment_research"


class VectorStore:
    """Persistent ChromaDB wrapper for investment research document chunks."""

    def __init__(
        self,
        persist_directory: Path | str | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        """Initialize the persistent ChromaDB client and collection."""
        path = Path(persist_directory) if persist_directory else chroma_db_dir()
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedding_model = embedding_model or get_embedding_model()

    def collection_count(self) -> int:
        """Return the number of chunks currently stored in the collection."""
        return self._collection.count()

    def add_documents(self, chunks: list[dict], force: bool = False) -> None:
        """Embed document chunks and add them to the vector store."""
        if not force and self.collection_count() > 0:
            logger.info("Vector store already populated, skipping ingestion")
            return

        if force and self.collection_count() > 0:
            self._client.delete_collection(COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        if not chunks:
            logger.warning("No document chunks provided for ingestion")
            return

        texts = [chunk["chunk_text"] for chunk in chunks]
        embeddings = self._embedding_model.embed_batch(texts)

        ids: list[str] = []
        metadatas: list[dict] = []
        documents: list[str] = []

        for index, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(
                f"{chunk['source_file']}:{index}:{chunk['chunk_text']}".encode("utf-8")
            ).hexdigest()
            ids.append(chunk_id)
            documents.append(chunk["chunk_text"])
            metadatas.append(
                {
                    "source_file": chunk["source_file"],
                    "company": chunk["company"],
                    "sector": chunk["sector"],
                    "doc_type": chunk["doc_type"],
                }
            )

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        sector_filter: str | None = None,
    ) -> list[dict]:
        """Search the vector store for similar document chunks."""
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if sector_filter:
            query_kwargs["where"] = {"sector": sector_filter}

        results = self._collection.query(**query_kwargs)

        matches: list[dict] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for document, metadata, distance in zip(documents, metadatas, distances):
            matches.append(
                {
                    "chunk_text": document,
                    "source_file": metadata.get("source_file"),
                    "company": metadata.get("company"),
                    "sector": metadata.get("sector"),
                    "doc_type": metadata.get("doc_type"),
                    "score": 1 - distance,
                }
            )

        return matches
