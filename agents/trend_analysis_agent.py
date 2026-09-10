"""Trend Analysis Agent definition."""

import logging
from collections import defaultdict

from llm.llm_router import LLMRouter
from rag.retriever import Retriever

logger = logging.getLogger(__name__)

TREND_ANALYSIS_SYNTHESIS_TEMPLATE = """You are a trend analysis specialist reviewing retrieved financial and news excerpts for an investment research demo.

User query: {query}
Sector filter: {sector}

Based ONLY on the retrieved context below, identify performance trends, comparative signals, and sector-wide risk patterns. Do not invent facts that are not supported by the excerpts.

Retrieved context grouped by company:
{company_context}

Return ONLY valid JSON with no markdown, code fences, or extra commentary.

The JSON must follow this structure exactly:
{{
  "sector_overview": "2-3 sentence summary of overall sector trends observed",
  "top_performers": [
    {{
      "company_name": "...",
      "reason": "why this company stands out positively"
    }}
  ],
  "underperformers_or_risks": [
    {{
      "company_name": "...",
      "reason": "why this company shows concerning signals"
    }}
  ],
  "comparative_insights": [
    "insight 1 comparing companies against each other",
    "insight 2",
    "insight 3"
  ],
  "sector_wide_risks": ["risk pattern seen across multiple companies", "..."]
}}
"""


class TrendAnalysisAgent:
    """Agent that retrieves relevant documents and synthesizes trend analysis."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm_router: LLMRouter | None = None,
    ) -> None:
        """Initialize the agent with retriever and LLM router instances."""
        self._retriever = retriever or Retriever()
        self._llm_router = llm_router or LLMRouter()

    def analyze(self, query: str, sector: str | None = None, top_k: int = 10) -> dict:
        """Retrieve relevant chunks and synthesize structured trend analysis.

        Args:
            query: Natural language trend analysis question.
            sector: Optional sector filter for retrieval.
            top_k: Number of chunks to retrieve.

        Returns:
            Parsed JSON trend analysis result from the LLM.

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
            raise ValueError(
                f"No relevant data found for trend analysis query: {query}"
            )

        grouped_chunks = self._group_chunks_by_company(chunks)
        company_context = self._format_grouped_chunks(grouped_chunks)
        sector_label = sector if sector else "all sectors"

        prompt = TREND_ANALYSIS_SYNTHESIS_TEMPLATE.format(
            query=query,
            sector=sector_label,
            company_context=company_context,
        )

        try:
            return self._llm_router.generate_json(prompt)
        except Exception as exc:
            logger.error(
                "Trend analysis synthesis failed for query='%s', sector=%r: %s",
                query,
                sector,
                exc,
            )
            raise RuntimeError(
                f"Trend analysis failed for query='{query}', sector={sector!r}: {exc}"
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
