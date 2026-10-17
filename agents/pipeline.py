"""Unified report generation pipeline with CrewAI and standalone engines."""

import json
import os
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.market_research_agent import MarketResearchAgent
from agents.summary_agent import SummaryAgent
from agents.trend_analysis_agent import TrendAnalysisAgent
from agents.orchestrator import run_crew
from rag.build_index import ensure_index

SUMMARY_REQUIRED_KEYS = (
    "report_title",
    "query",
    "executive_summary",
    "company_breakdown",
    "key_takeaways",
    "investment_considerations",
)


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def parse_summary_output(text: str) -> dict[str, Any]:
    """Parse and validate a Summary-schema JSON payload from LLM text."""
    cleaned = _strip_markdown_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Could not parse summary JSON from model output")
        data = json.loads(cleaned[start : end + 1])

    if not isinstance(data, dict):
        raise ValueError("Summary output must be a JSON object")

    missing = [key for key in SUMMARY_REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"Summary output missing keys: {', '.join(missing)}")

    considerations = data["investment_considerations"]
    if isinstance(considerations, list):
        data["investment_considerations"] = " ".join(str(item) for item in considerations)

    return data


def _package_result(
    report: dict[str, Any],
    sector: str | None,
    engine: str,
    started_at: float,
) -> dict[str, Any]:
    if not report.get("generated_at"):
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
    return {
        **report,
        "sector": sector,
        "engine": engine,
        "elapsed_seconds": round(time.time() - started_at, 2),
        "generated_at": report["generated_at"],
    }


def _run_standalone(query: str, sector: str | None, on_stage: Callable | None) -> dict[str, Any]:
    if on_stage:
        on_stage("analysis", "Running market research and trend analysis in parallel...")

    market_agent = MarketResearchAgent()
    trend_agent = TrendAnalysisAgent()
    summary_agent = SummaryAgent()

    with ThreadPoolExecutor(max_workers=2) as executor:
        market_future = executor.submit(market_agent.research, query, sector=sector)
        trend_future = executor.submit(trend_agent.analyze, query, sector=sector)
        market_result = market_future.result()
        trend_result = trend_future.result()

    if on_stage:
        on_stage("summary", "Synthesizing final report...")
    return summary_agent.summarize(
        market_research_output=market_result,
        trend_analysis_output=trend_result,
        original_query=query,
    )


def generate_report(
    query: str,
    sector: str | None = None,
    feedback: str | None = None,
    on_stage: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Generate an investment research report using the configured engine."""
    started_at = time.time()
    effective_query = query
    if feedback:
        effective_query = f"{query}\n\nReviewer feedback to address: {feedback}"

    if on_stage:
        on_stage("retrieval", "Ensuring knowledge base index is ready...")
    count = ensure_index()
    if count == 0:
        raise RuntimeError(
            "Knowledge base is empty. Add synthetic_data files and rebuild the index."
        )

    engine = os.getenv("ENGINE", "auto").strip().lower()
    errors: list[str] = []

    if engine in {"auto", "crewai"}:
        try:
            if on_stage:
                on_stage("analysis", "Running CrewAI orchestration...")
            raw = run_crew(effective_query, sector=sector)
            report = parse_summary_output(raw)
            if on_stage:
                on_stage("summary", "Report synthesized")
                on_stage("done", "Complete")
            return _package_result(report, sector, "crewai", started_at)
        except Exception as exc:
            errors.append(f"crewai: {exc}")
            if engine == "crewai":
                raise RuntimeError(
                    "CrewAI engine failed:\n" + "\n".join(errors)
                ) from exc

    if engine in {"auto", "standalone"}:
        try:
            report = _run_standalone(effective_query, sector, on_stage)
            if on_stage:
                on_stage("done", "Complete")
            return _package_result(report, sector, "standalone", started_at)
        except Exception as exc:
            errors.append(f"standalone: {exc}")

    raise RuntimeError("All engines failed:\n" + "\n".join(errors))
