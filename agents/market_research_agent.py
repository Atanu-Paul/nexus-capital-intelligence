"""Market Research Agent definition."""

import logging
from collections import defaultdict

from llm.llm_router import LLMRouter
from rag.retriever import Retriever

logger = logging.getLogger(__name__)

MARKET_RESEARCH_SYNTHESIS_TEMPLATE = """You are a market research analyst synthesizing retrieved financial and news excerpts for an investment research demo.

User query: {query}
Sector filter: {sector}

Based ONLY on the retrieved context below, produce a structured research summary. Do not invent facts that are not supported by the excerpts.

Retrieved context grouped by company:
{company_context}

Return ONLY valid JSON with no markdown, code fences, or extra commentary.

The JSON must follow this structure exactly:
{{
  "companies": [
    {{
      "company_name": "...",
      "sector": "...",
      "revenue_summary": "...",
      "growth_indicators": "...",
      "key_highlights": ["...", "..."],
      "risks_noted": ["...", "..."]
    }}
  ],
  "research_summary": "2-3 sentence overview of what was found across all companies in this query"
}}
"""


class MarketResearchAgent:
    """Agent that retrieves relevant documents and synthesizes market research."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm_router: LLMRouter | None = None,
    ) -> None:
        """Initialize the agent with retriever and LLM router instances."""
        self._retriever = retriever or Retriever()
        self._llm_router = llm_router or LLMRouter()

    def research(self, query: str, sector: str | None = None, top_k: int = 10) -> dict:
        """Retrieve relevant chunks and synthesize structured market research.

        Args:
            query: Natural language research question.
            sector: Optional sector filter for retrieval.
            top_k: Number of chunks to retrieve.

        Returns:
            Parsed JSON research result from the LLM.

        Raises:
            ValueError: If no relevant chunks are found.
            RuntimeError: If LLM synthesis fails after all provider fallbacks.
        """
        chunks = self._retriever.retrieve(
            query=query,
            top_k=top_k,
            sector_filter=sector,
        )

        if not chunks:
            raise ValueError(f"No relevant data found for query: {query}")

        grouped_chunks = self._group_chunks_by_company(chunks)
        company_context = self._format_grouped_chunks(grouped_chunks)
        sector_label = sector if sector else "all sectors"

        prompt = MARKET_RESEARCH_SYNTHESIS_TEMPLATE.format(
            query=query,
            sector=sector_label,
            company_context=company_context,
        )

        try:
            return self._llm_router.generate_json(prompt)
        except Exception as exc:
            logger.error(
                "Market research synthesis failed for query='%s', sector=%r: %s",
                query,
                sector,
                exc,
            )
            raise RuntimeError(
                f"Market research failed for query='{query}', sector={sector!r}: {exc}"
            ) from exc

    @staticmethod
    def _group_chunks_by_company(chunks: list[dict]) -> dict[str, list[dict]]:
        """Group retrieved chunks by company metadata."""
        grouped: dict[str, list[dict]] = defaultdict(list)
        for chunk in chunks:
            company = chunk.get("company") or "Unknown"
            grouped[company].append(chunk)
        return dict(grouped)

    @staticmethod
    def _format_grouped_chunks(grouped_chunks: dict[str, list[dict]]) -> str:
        """Format grouped chunks into prompt-ready text sections."""
        sections: list[str] = []

        for company, company_chunks in grouped_chunks.items():
            sector = company_chunks[0].get("sector", "unknown")
            excerpt_blocks = []

            for chunk in company_chunks:
                source_file = chunk.get("source_file", "unknown")
                doc_type = chunk.get("doc_type", "unknown")
                excerpt_blocks.append(
                    f"[{doc_type} | {source_file}]\n{chunk['chunk_text']}"
                )

            sections.append(
                f"Company: {company}\n"
                f"Sector: {sector}\n"
                f"Retrieved excerpts:\n" + "\n\n".join(excerpt_blocks)
            )

        return "\n\n---\n\n".join(sections)
