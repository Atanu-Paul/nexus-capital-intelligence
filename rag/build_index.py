"""Build the ChromaDB vector index from synthetic data."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.ingestion import load_all_documents
from rag.paths import synthetic_data_dir
from rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_index(force: bool = False) -> int:
    """Ensure the vector index exists; ingest synthetic data when empty.

    Args:
        force: Rebuild the collection even if it already has documents.

    Returns:
        Final collection chunk count.
    """
    store = VectorStore()
    count = store.collection_count()
    if force or count == 0:
        chunks = load_all_documents(synthetic_data_dir())
        logger.info("Loaded %s chunks from %s", len(chunks), synthetic_data_dir())
        store.add_documents(chunks, force=force)
    return store.collection_count()


def main() -> None:
    """Load synthetic documents and populate the vector store."""
    final_count = ensure_index(force=False)
    print(f"Final collection count: {final_count}")


if __name__ == "__main__":
    main()
