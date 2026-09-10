"""Manual test script for MarketResearchAgent."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.market_research_agent import MarketResearchAgent


def main() -> None:
    """Run a sample market research query and print the result."""
    agent = MarketResearchAgent()
    result = agent.research(
        "Compare revenue and growth across companies",
        sector="technology",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
