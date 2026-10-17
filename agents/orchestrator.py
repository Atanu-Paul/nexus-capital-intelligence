"""CrewAI-based orchestration for investment research agents."""

import logging
import sys
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.retrieval_tool import format_chunks
from rag.build_index import ensure_index
from rag.retrieval_service import retrieve_documents

load_dotenv()

logger = logging.getLogger(__name__)

primary_llm = LLM(model="gemini/gemini-3.5-flash-lite")
fallback_llm = LLM(model="groq/openai/gpt-oss-120b")


def _should_retry_with_fallback(exc: Exception) -> bool:
    """Return True only for provider/LLM failures (not data/index errors)."""
    if isinstance(exc, ValueError) and "No relevant data" in str(exc):
        return False
    if isinstance(exc, FileNotFoundError):
        return False
    return True


def build_agents(llm: LLM) -> dict[str, Agent]:
    """Build CrewAI agents for market research, trend analysis, and summary."""
    market_research_agent = Agent(
        role="Financial Data Researcher",
        goal=(
            "Gather and structure company financial data from the knowledge base "
            "for investment research queries."
        ),
        backstory=(
            "You are a detail-oriented equity research analyst who extracts accurate "
            "financial insights from retrieved documents and organizes them by company."
        ),
        tools=[],
        llm=llm,
        verbose=True,
    )

    trend_analysis_agent = Agent(
        role="Investment Trend Analyst",
        goal=(
            "Identify comparative performance patterns and trends across companies "
            "using retrieved financial and news data."
        ),
        backstory=(
            "You are an experienced market trend analyst skilled at spotting sector "
            "patterns, standout performers, and emerging risks across companies."
        ),
        tools=[],
        llm=llm,
        verbose=True,
    )

    summary_agent = Agent(
        role="Investment Report Writer",
        goal=(
            "Synthesize research and trend findings into a clear, neutral investment "
            "report without giving direct buy/sell financial advice."
        ),
        backstory=(
            "You are a professional financial report writer who produces balanced, "
            "informational summaries for analysts and never issues explicit trading "
            "recommendations."
        ),
        tools=[],
        llm=llm,
        verbose=True,
    )

    return {
        "market_research_agent": market_research_agent,
        "trend_analysis_agent": trend_analysis_agent,
        "summary_agent": summary_agent,
    }


def build_tasks(
    agents_dict: dict[str, Agent],
    query: str,
    sector: str | None,
    retrieved_context: str,
) -> list[Task]:
    """Build CrewAI tasks for parallel research/trend work and final summary."""
    sector_label = sector if sector else "all sectors"

    market_research_task = Task(
        description=(
            f"Research query: {query}\n"
            f"Sector filter: {sector_label}\n\n"
            "Use ONLY the pre-retrieved document context below. Do not call any tools.\n\n"
            f"Pre-retrieved context:\n{retrieved_context}\n\n"
            "Produce structured findings per company including revenue, growth, "
            "key highlights, and risks. Base your answer only on the context above."
        ),
        expected_output=(
            "Structured research findings per company with fields such as company_name, "
            "sector, revenue_summary, growth_indicators, key_highlights, and risks_noted, "
            "plus a short research_summary across all companies."
        ),
        agent=agents_dict["market_research_agent"],
        async_execution=True,
    )

    trend_analysis_task = Task(
        description=(
            f"Trend analysis query: {query}\n"
            f"Sector filter: {sector_label}\n\n"
            "Use ONLY the pre-retrieved document context below. Do not call any tools.\n\n"
            f"Pre-retrieved context:\n{retrieved_context}\n\n"
            "Produce comparative trend analysis including top performers, "
            "underperformers or risks, comparative insights, and sector-wide risk patterns."
        ),
        expected_output=(
            "Comparative trend analysis with sector_overview, top_performers, "
            "underperformers_or_risks, comparative_insights, and sector_wide_risks."
        ),
        agent=agents_dict["trend_analysis_agent"],
        async_execution=True,
    )

    summary_task = Task(
        description=(
            f"Original user query: {query}\n"
            f"Sector: {sector_label}\n\n"
            "Synthesize the market research and trend analysis outputs into a final "
            "investment research report. Include an executive summary, company "
            "breakdown, key takeaways, and neutral investment considerations. "
            "Do not provide direct buy/sell/hold recommendations."
        ),
        expected_output=(
            "A final report with report_title, query, executive_summary, "
            "company_breakdown, key_takeaways, and neutral investment_considerations."
        ),
        agent=agents_dict["summary_agent"],
        context=[market_research_task, trend_analysis_task],
        async_execution=False,
    )

    return [market_research_task, trend_analysis_task, summary_task]


def _prefetch_context(query: str, sector: str | None) -> str:
    """Retrieve document chunks once before the crew starts."""
    chunks = retrieve_documents(
        query=query,
        top_k=10,
        sector_filter=sector,
    )
    if not chunks:
        raise ValueError(f"No relevant data found for crew query: {query}")
    return format_chunks(chunks)


def _run_crew_with_llm(query: str, sector: str | None, llm: LLM) -> str:
    """Build and execute a crew with the provided LLM."""
    ensure_index()
    retrieved_context = _prefetch_context(query, sector)
    agents_dict = build_agents(llm)
    tasks = build_tasks(agents_dict, query, sector, retrieved_context)

    crew = Crew(
        agents=list(agents_dict.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)


def run_crew(query: str, sector: str | None = None) -> str:
    """Run the investment research crew with Gemini, falling back to Groq on LLM failure."""
    primary_error: Exception | None = None

    try:
        return _run_crew_with_llm(query, sector, primary_llm)
    except Exception as exc:
        if not _should_retry_with_fallback(exc):
            raise
        primary_error = exc
        logger.warning(
            "Primary LLM (Gemini) failed: %s. Retrying full crew with Groq fallback...",
            exc,
        )

    try:
        return _run_crew_with_llm(query, sector, fallback_llm)
    except Exception as fallback_exc:
        raise RuntimeError(
            "Crew execution failed for both primary Gemini and Groq fallback LLMs.\n"
            f"Primary error: {primary_error}\n"
            f"Fallback error: {fallback_exc}"
        ) from fallback_exc
