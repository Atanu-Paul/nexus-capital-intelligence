"""Manual test script for the CrewAI orchestrator."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestrator import run_crew


def main() -> None:
    """Run a sample crew workflow and print the final result."""
    result = run_crew(
        "Compare technology sector companies Q3 performance",
        sector="technology",
    )
    print(result)


if __name__ == "__main__":
    main()
