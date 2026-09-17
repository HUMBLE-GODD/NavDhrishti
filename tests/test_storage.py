"""
Unit tests for SQLite Audit Storage Manager.
"""

from pathlib import Path
import pytest

from navdhrishti.detector import DetectedDefect, InspectionResult
from navdhrishti.storage import AuditStorageManager


@pytest.fixture
def temp_db(tmp_path: Path) -> AuditStorageManager:
    db_file = tmp_path / "test_audit.db"
    return AuditStorageManager(db_file)


def test_schema_initialization(temp_db: AuditStorageManager) -> None:
    assert temp_db.db_path.exists()
    with temp_db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        assert "inspections" in tables
        assert "defect_records" in tables
        assert "system_metrics" in tables


def test_log_and_query_inspection(temp_db: AuditStorageManager) -> None:
    sample_defect = DetectedDefect(
        defect_id=1,
        defect_type="crack",
        confidence=0.92,
        bbox=(20, 30, 40, 50),
        area=340.0,
        severity=72.5,
    )
    result = InspectionResult(
        image_path="/dummy/test_crack.png",
        overall_status="REJECT",
        defect_count=1,
        max_severity=72.5,
        mean_severity=72.5,
        defects=[sample_defect],
        inference_time_ms=45.2,
        binary_mask=None,
    )

    insp_id = temp_db.log_inspection(result)
    assert insp_id == 1

    records = temp_db.query_recent_inspections(limit=5)
    assert len(records) == 1
    assert records[0]["overall_status"] == "REJECT"
    assert records[0]["defect_count"] == 1
    assert records[0]["max_severity"] == 72.5

    stats = temp_db.get_summary_statistics()
    assert stats["total_inspections"] == 1
    assert stats["status_breakdown"]["REJECT"] == 1
    assert stats["defect_breakdown"]["crack"] == 1
