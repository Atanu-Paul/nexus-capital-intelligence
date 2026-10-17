"""Streamlit entry point - streamlit run app/main.py"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from agents.pipeline import generate_report
from app.components.report_display import render_report
from app.components.sidebar import render_sidebar
from app.utils.session_state import init_session_state, set_stage, update_status_line
from deployment.s3_upload import S3Storage
from human_approval.approval_flow import ReportStore


@st.cache_resource
def get_report_store() -> ReportStore:
    return ReportStore(s3_storage=get_s3_storage())


@st.cache_resource
def get_s3_storage() -> S3Storage:
    return S3Storage()


def _friendly_error(exc: Exception) -> tuple[str, str]:
    message = str(exc)
    lower = message.lower()
    if "knowledge base is empty" in lower or "no relevant data" in lower:
        return (
            "No indexed data found for this query. Rebuild the knowledge base and try again.",
            message,
        )
    if "quota" in lower or "429" in lower or "all engines failed" in lower:
        return (
            "All configured models appear quota-exhausted or unavailable. Try again later or switch engine.",
            message,
        )
    return ("Something went wrong while generating the report.", message)


def _run_research(query: str, sector: str | None) -> None:
    def on_stage(stage: str, message: str) -> None:
        update_status_line(stage, message)

    try:
        report = generate_report(query=query, sector=sector, on_stage=on_stage)
        record = get_report_store().create_draft(report=report, query=query, sector=sector)
        st.session_state["last_report"] = report
        st.session_state["current_record"] = record
        st.session_state["error_message"] = None
        st.session_state["error_detail"] = None
        set_stage("pending_approval")
    except Exception as exc:
        friendly, detail = _friendly_error(exc)
        st.session_state["error_message"] = friendly
        st.session_state["error_detail"] = detail
        set_stage("idle")


def main() -> None:
    st.set_page_config(page_title="Nexus Capital Intelligence", layout="wide")
    init_session_state()

    st.title("Nexus Capital Intelligence")
    st.warning(
        "All company data is synthetic. Output is informational and not investment advice."
    )

    sector, engine = render_sidebar()
    os.environ["ENGINE"] = engine

    research_tab, approved_tab = st.tabs(["Research", "Approved reports"])

    with research_tab:
        st.session_state["query"] = st.text_area(
            "Research query",
            value=st.session_state.get("query", ""),
            height=120,
            key="research_query",
        )
        query = st.session_state["query"].strip()
        run_disabled = not query or st.session_state["stage"] == "approved"

        if st.button("Run research", disabled=run_disabled, key="run_research"):
            set_stage("idle")
            with st.status("Running research pipeline...", expanded=True) as status:
                _run_research(query, sector)
                status.update(label="Research pipeline finished", state="complete")

        if st.session_state.get("error_message"):
            st.error(st.session_state["error_message"])
            with st.expander("Technical details"):
                st.code(st.session_state.get("error_detail", ""))

        record = st.session_state.get("current_record")
        report = st.session_state.get("last_report")
        if report:
            draft_prefix = f"draft_{record['report_id']}" if record else "draft_live"
            render_report(report, key_prefix=draft_prefix, record=record)

        if record and st.session_state["stage"] == "pending_approval":
            st.markdown("### Human approval")
            reviewer = st.text_input("Reviewer name", key="reviewer_name")
            comment = st.text_area("Reviewer comment", key="reviewer_comment")
            col1, col2, col3 = st.columns(3)
            if col1.button("Approve & publish", key="approve_publish"):
                if not reviewer:
                    st.error("Reviewer name is required.")
                else:
                    updated = get_report_store().approve(
                        record["report_id"], reviewer=reviewer, comment=comment
                    )
                    st.session_state["current_record"] = updated
                    set_stage("approved")
                    st.rerun()
            if col2.button("Reject", key="reject_report"):
                if not reviewer:
                    st.error("Reviewer name is required.")
                else:
                    updated = get_report_store().reject(
                        record["report_id"], reviewer=reviewer, comment=comment
                    )
                    st.session_state["current_record"] = updated
                    set_stage("rejected")
                    st.rerun()
            if col3.button("Reject & revise", key="reject_revise"):
                if not reviewer:
                    st.error("Reviewer name is required.")
                else:
                    get_report_store().reject(
                        record["report_id"], reviewer=reviewer, comment=comment
                    )
                    revised = generate_report(
                        query=query,
                        sector=sector,
                        feedback=comment or "Please revise the report.",
                    )
                    updated = get_report_store().add_revision(
                        record["report_id"], revised
                    )
                    st.session_state["last_report"] = revised
                    st.session_state["current_record"] = updated
                    set_stage("pending_approval")
                    st.rerun()

        if record and st.session_state["stage"] == "approved":
            if record.get("s3_saved"):
                st.success(f"Published to S3: {record.get('s3_key')}")
            else:
                st.warning(
                    f"Saved locally only. S3 error: {record.get('s3_error', 'unknown')}"
                )

    with approved_tab:
        approved_reports = get_report_store().list_approved()
        if not approved_reports:
            st.info("No approved reports yet.")
        else:
            labels = [
                f"{item.get('report_id', 'n/a')} - {item.get('report', {}).get('report_title', 'Report')}"
                for item in approved_reports
            ]
            selected = st.selectbox("Select report", labels, key="approved_report_select")
            index = labels.index(selected)
            chosen = approved_reports[index]
            chosen_id = chosen.get("report_id", f"idx_{index}")
            render_report(
                chosen.get("report", {}),
                key_prefix=f"approved_{chosen_id}",
                read_only=True,
            )


if __name__ == "__main__":
    main()
