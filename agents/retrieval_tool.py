"""CrewAI tool wrapping the thread-safe RAG retrieval service."""

import sys
from pathlib import Path

from crewai.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.retrieval_service import retrieve_documents


def format_chunks(chunks: list[dict]) -> str:
    """Format retrieved chunks as readable text for CrewAI agents."""
    if not chunks:
        return "No relevant documents found for this query."

    sections: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        company = chunk.get("company", "Unknown")
        doc_type = chunk.get("doc_type", "unknown")
        source_file = chunk.get("source_file", "unknown")
        score = chunk.get("score")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"

        sections.append(
            f"Result {index}\n"
            f"Company: {company}\n"
            f"Document type: {doc_type}\n"
            f"Source file: {source_file}\n"
            f"Similarity score: {score_text}\n"
            f"Content:\n{chunk.get('chunk_text', '')}"
        )

    return "\n\n---\n\n".join(sections)


@tool("financial_data_retriever")
def financial_data_retriever(query: str, sector: str = "") -> str:
    """Retrieve financial research document chunks for a query and optional sector filter."""
    sector_filter = sector.strip() or None
    chunks = retrieve_documents(
        query=query,
        top_k=10,
        sector_filter=sector_filter,
    )
    return format_chunks(chunks)
