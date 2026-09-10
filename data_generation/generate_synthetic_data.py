"""Script to generate synthetic reports/news via LLM."""

import os
import re
import sys
import time
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.prompt_templates import NEWS_PROMPT_TEMPLATE, REPORT_PROMPT_TEMPLATE
from llm.llm_router import LLMRouter

QUARTER = "Q3 2026"

COMPANIES = {
    "technology": [
        {"name": "NovaTech Systems", "ticker": "NVTC"},
        {"name": "Quantix Software", "ticker": "QNTX"},
        {"name": "Orbital Cloud Inc.", "ticker": "ORBC"},
        {"name": "Vertex Semiconductors", "ticker": "VRTX"},
        {"name": "Lumen Data Corp", "ticker": "LMDC"},
    ],
    "banking": [
        {"name": "Zenith National Bank", "ticker": "ZNTH"},
        {"name": "Meridian Trust Bank", "ticker": "MRDT"},
        {"name": "Ashford Financial Group", "ticker": "ASHF"},
        {"name": "Highline Capital Bank", "ticker": "HLCB"},
        {"name": "Sterling Reserve Bank", "ticker": "STRB"},
    ],
    "pharma": [
        {"name": "Cascade Biopharma", "ticker": "CSBP"},
        {"name": "Novagen Therapeutics", "ticker": "NVGT"},
        {"name": "Harborview Pharmaceuticals", "ticker": "HRBV"},
        {"name": "Elysium Life Sciences", "ticker": "ELSC"},
        {"name": "Pinnacle Health Corp", "ticker": "PNHC"},
    ],
}


def company_slug(name: str) -> str:
    """Convert a company name to a filesystem-safe slug."""
    slug = name.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s]+", "_", slug.strip())
    return slug


def make_pdf(filepath: str, title: str, content_dict: dict) -> None:
    """Render a dict's contents into a simple structured PDF."""
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]

    for key, value in content_dict.items():
        heading = str(key).replace("_", " ").title()
        story.append(Paragraph(heading, styles["Heading2"]))

        if isinstance(value, list):
            for item in value:
                story.append(Paragraph(f"&bull; {item}", styles["Normal"]))
        else:
            story.append(Paragraph(str(value), styles["Normal"]))

        story.append(Spacer(1, 8))

    doc.build(story)


def save_news(filepath: Path, news_data: dict) -> None:
    """Save news snippets to a plain text file."""
    lines = []
    for item in news_data["news"]:
        lines.append(f"HEADLINE: {item['headline']}")
        lines.append(f"DATE: {item['date']}")
        lines.append(f"SENTIMENT: {item['sentiment']}")
        lines.append(item["body"])
        lines.append("")

    filepath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Generate synthetic report PDFs and news text files for all companies."""
    client = LLMRouter()
    succeeded = 0
    failed = 0
    synthetic_data_dir = PROJECT_ROOT / "synthetic_data"

    for sector, companies in COMPANIES.items():
        output_dir = synthetic_data_dir / sector
        output_dir.mkdir(parents=True, exist_ok=True)

        for company in companies:
            name = company["name"]
            ticker = company["ticker"]
            slug = company_slug(name)
            company_failed = False

            report_path = output_dir / f"{slug}_q3report.pdf"
            news_path = output_dir / f"{slug}_news.txt"

            print(f"Generating data for {name} ({sector})...")

            if os.path.exists(report_path):
                print(f"Skipping {name} report - already exists")
            else:
                try:
                    report_prompt = REPORT_PROMPT_TEMPLATE.format(
                        company_name=name,
                        sector=sector,
                        ticker=ticker,
                        quarter=QUARTER,
                    )
                    report_data = client.generate_json(report_prompt)
                    make_pdf(str(report_path), f"{name} - {QUARTER} Report", report_data)
                except Exception as exc:
                    print(f"  ERROR generating report for {name}: {exc}")
                    company_failed = True

                time.sleep(2)

            if os.path.exists(news_path):
                print(f"Skipping {name} news - already exists")
            else:
                try:
                    news_prompt = NEWS_PROMPT_TEMPLATE.format(
                        company_name=name,
                        sector=sector,
                        quarter=QUARTER,
                    )
                    news_data = client.generate_json(news_prompt)
                    save_news(news_path, news_data)
                except Exception as exc:
                    print(f"  ERROR generating news for {name}: {exc}")
                    company_failed = True

                time.sleep(2)

            if company_failed:
                failed += 1
            else:
                succeeded += 1

    print(f"\nSummary: {succeeded} companies succeeded, {failed} companies failed.")


if __name__ == "__main__":
    main()
