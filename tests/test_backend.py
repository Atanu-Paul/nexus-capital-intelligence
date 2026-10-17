"""Backend tests without network or live LLM calls."""

import json
import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sample_summary() -> dict:
    return {
        "report_title": "Test Report",
        "query": "Test query",
        "executive_summary": "Summary text.",
        "company_breakdown": [{"company_name": "Acme", "summary": "Doing fine."}],
        "key_takeaways": ["One", "Two", "Three"],
        "investment_considerations": "Neutral considerations only.",
        "generated_at": "2026-01-01T00:00:00+00:00",
    }


def test_retrieval_independent_of_cwd(tmp_path, monkeypatch):
    from rag.build_index import ensure_index
    from rag.retrieval_service import retrieve_documents

    ensure_index()
    monkeypatch.chdir(tmp_path)
    chunks = retrieve_documents(
        "technology revenue growth",
        top_k=3,
        sector_filter="technology",
        use_cache=False,
    )
    assert chunks


class _FakeS3:
    def available(self) -> bool:
        return False


def test_report_store_local_workflow(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "human_approval.approval_flow.REPORTS_DIR",
        tmp_path / "reports",
    )
    monkeypatch.setattr(
        "human_approval.approval_flow.LOCAL_APPROVED_DIR",
        tmp_path / "approved",
    )

    from human_approval.approval_flow import ReportStore

    store = ReportStore(s3_storage=_FakeS3())
    report = _sample_summary()
    draft = store.create_draft(report, query="Q", sector="technology")
    assert draft["status"] == "pending_approval"

    approved = store.approve(draft["report_id"], reviewer="alice", comment="ok")
    assert approved["status"] == "approved"
    assert approved["s3_saved"] is False

    revised = store.add_revision(draft["report_id"], _sample_summary())
    assert revised["revision"] == 2
    assert revised["status"] == "pending_approval"

    rejected = store.reject(draft["report_id"], reviewer="bob", comment="no")
    assert rejected["status"] == "rejected"


def test_pipeline_falls_back_to_standalone(monkeypatch):
    monkeypatch.setenv("ENGINE", "auto")
    monkeypatch.setattr("agents.pipeline.ensure_index", lambda force=False: 10)

    def fail_crew(query, sector=None):
        raise RuntimeError("crew failed")

    monkeypatch.setattr("agents.pipeline.run_crew", fail_crew)
    monkeypatch.setattr(
        "agents.pipeline._run_standalone",
        lambda query, sector, on_stage: _sample_summary(),
    )

    from agents.pipeline import generate_report

    result = generate_report("Compare companies", sector="technology")
    assert result["engine"] == "standalone"
    assert result["report_title"] == "Test Report"


def test_orchestrator_does_not_retry_on_no_data(monkeypatch):
    calls = {"count": 0}

    def fake_run(query, sector, llm):
        calls["count"] += 1
        raise ValueError("No relevant data found for crew query: missing")

    monkeypatch.setattr("agents.orchestrator.ensure_index", lambda force=False: 1)
    monkeypatch.setattr("agents.orchestrator._run_crew_with_llm", fake_run)

    from agents.orchestrator import run_crew

    with pytest.raises(ValueError):
        run_crew("missing")
    assert calls["count"] == 1


def test_streamlit_app_loads():
    from streamlit.testing.v1 import AppTest

    app_path = PROJECT_ROOT / "app" / "main.py"
    app = AppTest.from_file(str(app_path))
    app.run(timeout=60)
    assert not app.exception


def test_streamlit_duplicate_report_render_no_exception(monkeypatch):
    from streamlit.testing.v1 import AppTest

    report_id = "abc12345"
    report = _sample_summary()
    approved_item = {
        "report_id": report_id,
        "query": report["query"],
        "sector": "technology",
        "revision": 1,
        "reviewer": "alice",
        "comment": "ok",
        "approved_at": "2026-01-01T00:00:00+00:00",
        "report": report,
    }
    record = {
        "report_id": report_id,
        "status": "approved",
        "query": report["query"],
        "sector": "technology",
        "revision": 1,
        "report": report,
        "history": [],
        "reviewer": "alice",
        "comment": "ok",
        "created_at": "2026-01-01T00:00:00+00:00",
        "decided_at": "2026-01-01T00:00:00+00:00",
        "s3_key": None,
        "s3_saved": False,
        "s3_error": "S3 unavailable",
    }

    class _FakeStore:
        def list_approved(self):
            return [approved_item]

    monkeypatch.setattr("rag.build_index.ensure_index", lambda force=False: 5)
    monkeypatch.setattr("app.main.get_report_store", lambda: _FakeStore())
    monkeypatch.setattr(
        "deployment.s3_upload.S3Storage.available",
        lambda self: False,
    )

    app_path = PROJECT_ROOT / "app" / "main.py"
    app = AppTest.from_file(str(app_path))
    app.session_state["last_report"] = report
    app.session_state["current_record"] = record
    app.session_state["stage"] = "approved"
    app.session_state["query"] = report["query"]
    app.run(timeout=60)
    assert not app.exception
