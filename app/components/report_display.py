"""Report display UI component."""

import json
from typing import Any

import streamlit as st


def _report_to_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report.get('report_title', 'Investment Research Report')}",
        "",
        f"**Query:** {report.get('query', '')}",
        "",
        "## Executive Summary",
        str(report.get("executive_summary", "")),
        "",
        "## Company Breakdown",
    ]
    for company in report.get("company_breakdown", []):
        name = company.get("company_name", "Unknown")
        summary = company.get("summary") or company.get("revenue_summary", "")
        lines.extend([f"### {name}", str(summary), ""])
    lines.extend(["## Key Takeaways"])
    for item in report.get("key_takeaways", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Investment Considerations",
            str(report.get("investment_considerations", "")),
        ]
    )
    return "\n".join(lines)


def render_report(
    report: dict[str, Any],
    key_prefix: str,
    record: dict[str, Any] | None = None,
    read_only: bool = False,
) -> None:
    """Render a generated or approved report."""
    st.subheader(report.get("report_title", "Investment Research Report"))
    cols = st.columns(4)
    cols[0].metric("Engine", report.get("engine", record.get("engine", "n/a") if record else "n/a"))
    cols[1].metric(
        "Elapsed (s)",
        report.get("elapsed_seconds", "n/a"),
    )
    cols[2].metric(
        "Revision",
        record.get("revision", 1) if record else report.get("revision", 1),
    )
    cols[3].metric(
        "Status",
        record.get("status", "generated") if record else report.get("status", "generated"),
    )

    st.markdown("### Executive Summary")
    st.write(report.get("executive_summary", ""))

    for index, company in enumerate(report.get("company_breakdown", [])):
        title = company.get("company_name", "Company")
        with st.expander(title, expanded=False):
            if company.get("summary"):
                st.write(company["summary"])
            else:
                for key, value in company.items():
                    if key == "company_name":
                        continue
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")

    st.markdown("### Key Takeaways")
    for item in report.get("key_takeaways", []):
        st.markdown(f"- {item}")

    st.info(report.get("investment_considerations", ""))

    json_payload = json.dumps(report, indent=2)
    st.download_button(
        "Download JSON",
        data=json_payload,
        file_name="report.json",
        mime="application/json",
        disabled=read_only and False,
        key=f"{key_prefix}_dl_json",
    )
    st.download_button(
        "Download Markdown",
        data=_report_to_markdown(report),
        file_name="report.md",
        mime="text/markdown",
        key=f"{key_prefix}_dl_md",
    )
