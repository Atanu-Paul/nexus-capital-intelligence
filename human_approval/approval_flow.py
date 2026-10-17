"""Human approval checkpoint logic."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deployment.s3_upload import S3Storage
from rag.paths import PROJECT_ROOT

REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
LOCAL_APPROVED_DIR = PROJECT_ROOT / "data" / "approved"


class ReportStore:
    """JSON-file backed report drafts and approval workflow."""

    def __init__(self, s3_storage: S3Storage | None = None) -> None:
        self._s3 = s3_storage or S3Storage()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        LOCAL_APPROVED_DIR.mkdir(parents=True, exist_ok=True)

    def create_draft(self, report: dict, query: str, sector: str | None) -> dict[str, Any]:
        """Create a pending approval record for a generated report."""
        report_id = uuid.uuid4().hex[:8]
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "report_id": report_id,
            "status": "pending_approval",
            "query": query,
            "sector": sector,
            "revision": 1,
            "report": report,
            "history": [],
            "reviewer": None,
            "comment": "",
            "created_at": now,
            "decided_at": None,
            "s3_key": None,
            "s3_saved": False,
            "s3_error": None,
        }
        self._save(record)
        return record

    def get(self, report_id: str) -> dict[str, Any]:
        """Load a report record by id."""
        path = self._record_path(report_id)
        if not path.exists():
            raise KeyError(f"Report not found: {report_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def approve(self, report_id: str, reviewer: str, comment: str = "") -> dict[str, Any]:
        """Approve a report and publish to S3 when available."""
        record = self.get(report_id)
        now = datetime.now(timezone.utc).isoformat()
        record["status"] = "approved"
        record["reviewer"] = reviewer
        record["comment"] = comment
        record["decided_at"] = now

        date_prefix = now[:10]
        s3_key = (
            f"approved_reports/{date_prefix}/"
            f"{report_id}_v{record['revision']}.json"
        )
        payload = {
            "report_id": report_id,
            "query": record["query"],
            "sector": record["sector"],
            "revision": record["revision"],
            "reviewer": reviewer,
            "comment": comment,
            "approved_at": now,
            "report": record["report"],
        }

        local_path = LOCAL_APPROVED_DIR / f"{report_id}_v{record['revision']}.json"
        local_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        record["s3_key"] = s3_key
        if self._s3.available():
            try:
                self._s3.upload_json(s3_key, payload)
                record["s3_saved"] = True
                record["s3_error"] = None
            except Exception as exc:
                record["s3_saved"] = False
                record["s3_error"] = str(exc)
        else:
            record["s3_saved"] = False
            record["s3_error"] = "S3 unavailable"

        self._save(record)
        return record

    def reject(self, report_id: str, reviewer: str, comment: str = "") -> dict[str, Any]:
        """Reject a report."""
        record = self.get(report_id)
        record["status"] = "rejected"
        record["reviewer"] = reviewer
        record["comment"] = comment
        record["decided_at"] = datetime.now(timezone.utc).isoformat()
        self._save(record)
        return record

    def add_revision(self, report_id: str, new_report: dict) -> dict[str, Any]:
        """Store a revised report and reset to pending approval."""
        record = self.get(report_id)
        record["history"].append(
            {
                "revision": record["revision"],
                "report": record["report"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        record["revision"] += 1
        record["report"] = new_report
        record["status"] = "pending_approval"
        record["reviewer"] = None
        record["comment"] = ""
        record["decided_at"] = None
        record["s3_key"] = None
        record["s3_saved"] = False
        record["s3_error"] = None
        self._save(record)
        return record

    def list_approved(self) -> list[dict[str, Any]]:
        """List approved reports from S3 when available, otherwise local files."""
        if self._s3.available():
            try:
                keys = [
                    key
                    for key in self._s3.list_keys("approved_reports/")
                    if key.endswith(".json")
                ]
                reports = [self._s3.download_json(key) for key in sorted(keys)]
                return reports
            except Exception:
                pass

        reports: list[dict[str, Any]] = []
        for path in sorted(LOCAL_APPROVED_DIR.glob("*.json")):
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        return reports

    def _record_path(self, report_id: str) -> Path:
        return REPORTS_DIR / f"{report_id}.json"

    def _save(self, record: dict[str, Any]) -> None:
        self._record_path(record["report_id"]).write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )
