"""Summary Agent definition."""

import json
import logging
from datetime import datetime, timezone

from llm.llm_router import LLMRouter

logger = logging.getLogger(__name__)

SUMMARY_SYNTHESIS_TEMPLATE = """You are an investment research report writer synthesizing outputs from a market research agent and a trend analysis agent for an academic demo.

Original user query: {original_query}

Market research output (JSON):
{market_research_json}

Trend analysis output (JSON):
{trend_analysis_json}

Based ONLY on the inputs above, produce a cohesive investment research summary. Do not invent facts that are not supported by the provided agent outputs.

Return ONLY valid JSON with no markdown, code fences, or extra commentary.

The JSON must follow this structure exactly:
{{
  "report_title": "a professional title for this investment research report",
  "query": "the original query, restated",
  "executive_summary": "3-4 sentence high-level summary combining research and trend findings",
  "company_breakdown": [
    {{
      "company_name": "...",
      "summary": "1-2 sentence synthesis combining that company's research data and any trend analysis mentions of it"
    }}
  ],
  "key_takeaways": ["takeaway 1", "takeaway 2", "takeaway 3"],
  "investment_considerations": "2-3 sentences of balanced considerations - things an analyst should weigh, phrased neutrally, NOT as direct buy/sell advice",
  "generated_at": ""
}}

Important: Keep investment_considerations neutral and informational only. Do not provide direct financial advice or explicit buy/sell/hold recommendations.
"""


class SummaryAgent:
    """Agent that synthesizes market research and trend analysis into a final report.

    The ``investment_considerations`` field must remain neutral and informational.
    Output must never be phrased as direct financial advice (for example, explicit
    buy, sell, or hold recommendations). This supports responsible-AI practice and
    the downstream human approval step.
    """

    def __init__(self, llm_router: LLMRouter | None = None) -> None:
        """Initialize the agent with an LLM router instance."""
        self._llm_router = llm_router or LLMRouter()

    def summarize(
        self,
        market_research_output: dict,
        trend_analysis_output: dict,
        original_query: str,
    ) -> dict:
        """Synthesize market research and trend analysis into a final report.

        This method does not retrieve additional data. It only synthesizes the
        provided agent outputs.

        Args:
            market_research_output: Output from ``MarketResearchAgent.research()``.
            trend_analysis_output: Output from ``TrendAnalysisAgent.analyze()``.
            original_query: The user's original research question.

        Returns:
            Parsed JSON summary report with ``generated_at`` set in code.

        Raises:
            RuntimeError: If LLM synthesis fails after all provider fallbacks.
        """
        prompt = SUMMARY_SYNTHESIS_TEMPLATE.format(
            original_query=original_query,
            market_research_json=json.dumps(market_research_output, indent=2),
            trend_analysis_json=json.dumps(trend_analysis_output, indent=2),
        )

        try:
            result = self._llm_router.generate_json(prompt)
        except Exception as exc:
            logger.error(
                "Summary synthesis failed for query='%s': %s",
                original_query,
                exc,
            )
            raise RuntimeError(
                f"Summary synthesis failed for query='{original_query}': {exc}"
            ) from exc

        result["generated_at"] = datetime.now(timezone.utc).isoformat()
        return result
