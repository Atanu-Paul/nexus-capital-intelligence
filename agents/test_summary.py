"""Manual test script for SummaryAgent with upstream agents."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.market_research_agent import MarketResearchAgent
from agents.summary_agent import SummaryAgent
from agents.trend_analysis_agent import TrendAnalysisAgent


def main() -> None:
    """Run research, trend analysis, and summary synthesis for a sample query."""
    query = "Compare technology sector companies Q3 performance"
    sector = "technology"

    market_agent = MarketResearchAgent()
    trend_agent = TrendAnalysisAgent()
    summary_agent = SummaryAgent()

    market_result = market_agent.research(query, sector=sector)
    trend_result = trend_agent.analyze(query, sector=sector)
    summary_result = summary_agent.summarize(
        market_research_output=market_result,
        trend_analysis_output=trend_result,
        original_query=query,
    )

    print(json.dumps(summary_result, indent=2))


if __name__ == "__main__":
    main()
