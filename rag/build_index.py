"""Build the ChromaDB vector index from synthetic data."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.ingestion import load_all_documents
from rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)


def main() -> None:
    """Load synthetic documents and populate the vector store."""
    chunks = load_all_documents()
    print(f"Loaded {len(chunks)} chunks from synthetic_data/")

    store = VectorStore()
    store.add_documents(chunks)
    print(f"Final collection count: {store.collection_count()}")


if __name__ == "__main__":
    main()
