"""Manual test script for TrendAnalysisAgent."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.trend_analysis_agent import TrendAnalysisAgent


def main() -> None:
    """Run a sample trend analysis query and print the result."""
    agent = TrendAnalysisAgent()
    result = agent.analyze(
        "Compare performance trends across companies",
        sector="technology",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
