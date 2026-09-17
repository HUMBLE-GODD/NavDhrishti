"""
Relational Storage & Audit Trail Manager.
Uses SQLite for zero-configuration, robust persistence of inspection records,
localized defect coordinates, and operational telemetry.
"""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import sqlite3
from typing import Dict, Generator, List, Optional, Union

from navdhrishti.config import InspectionConfig
from navdhrishti.detector import InspectionResult


class AuditStorageManager:
    """Manages the SQLite database schema and audit log persistence."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        if db_path is None:
            config = InspectionConfig()
            self.db_path = config.db_path
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Create database connection with foreign key enforcement and automatic closing."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialize relational database tables and performance indexes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Table 1: Master Inspections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inspections (
                    inspection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    image_path TEXT NOT NULL,
                    image_hash TEXT,
                    overall_status TEXT CHECK(overall_status IN ('PASS', 'WARNING', 'REJECT')),
                    defect_count INTEGER NOT NULL,
                    max_severity REAL NOT NULL,
                    mean_severity REAL NOT NULL,
                    inference_time_ms REAL NOT NULL
                );
            """)

            # Table 2: Child Defect records table (1-to-many relationship)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS defect_records (
                    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection_id INTEGER NOT NULL,
                    defect_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_x INTEGER NOT NULL,
                    bbox_y INTEGER NOT NULL,
                    bbox_w INTEGER NOT NULL,
                    bbox_h INTEGER NOT NULL,
                    area_px REAL NOT NULL,
                    severity_score REAL NOT NULL,
                    FOREIGN KEY (inspection_id) REFERENCES inspections(inspection_id) ON DELETE CASCADE
                );
            """)

            # Table 3: System Performance and Batch Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_name TEXT NOT NULL,
                    total_inspected INTEGER NOT NULL,
                    pass_count INTEGER NOT NULL,
                    warning_count INTEGER NOT NULL,
                    reject_count INTEGER NOT NULL,
                    mean_fps REAL NOT NULL
                );
            """)

            # Indexes for fast lookup
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_insp_status ON inspections(overall_status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_insp_time ON inspections(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_defect_insp ON defect_records(inspection_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_defect_type ON defect_records(defect_type);")
            conn.commit()

    def calculate_file_hash(self, file_path: Union[str, Path]) -> str:
        """Compute SHA-256 checksum of the inspected image."""
        path = Path(file_path)
        if not path.exists():
            return "unknown_hash"
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def log_inspection(self, result: InspectionResult) -> int:
        """
        Persist an inspection result and all localized defect records atomically.
        
        Returns:
            int: Assigned inspection_id primary key.
        """
        img_hash = self.calculate_file_hash(result.image_path)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO inspections (
                    image_path, image_hash, overall_status, defect_count,
                    max_severity, mean_severity, inference_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.image_path,
                    img_hash,
                    result.overall_status,
                    result.defect_count,
                    result.max_severity,
                    result.mean_severity,
                    result.inference_time_ms,
                ),
            )
            inspection_id = cursor.lastrowid

            for defect in result.defects:
                x, y, w, h = defect.bbox
                cursor.execute(
                    """
                    INSERT INTO defect_records (
                        inspection_id, defect_type, confidence,
                        bbox_x, bbox_y, bbox_w, bbox_h, area_px, severity_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        inspection_id,
                        defect.defect_type,
                        defect.confidence,
                        x,
                        y,
                        w,
                        h,
                        defect.area,
                        defect.severity,
                    ),
                )

            conn.commit()
            return int(inspection_id)

    def log_batch_session(
        self, session_name: str, total: int, passes: int, warnings: int, rejects: int, mean_fps: float
    ) -> int:
        """Log summary metrics for a batch inspection run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO system_metrics (
                    session_name, total_inspected, pass_count, warning_count, reject_count, mean_fps
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_name, total, passes, warnings, rejects, mean_fps),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def query_recent_inspections(self, limit: int = 10) -> List[Dict]:
        """Fetch the most recent inspection records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT inspection_id, timestamp, image_path, overall_status,
                       defect_count, max_severity, inference_time_ms
                FROM inspections
                ORDER BY inspection_id DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_summary_statistics(self) -> Dict:
        """Aggregate statistical distribution of defect classes and overall quality statuses."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(inference_time_ms) FROM inspections")
            total_inspections, avg_latency = cursor.fetchone()

            cursor.execute(
                "SELECT overall_status, COUNT(*) FROM inspections GROUP BY overall_status"
            )
            status_dist = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute(
                "SELECT defect_type, COUNT(*) FROM defect_records GROUP BY defect_type"
            )
            defect_dist = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "total_inspections": total_inspections or 0,
                "avg_latency_ms": round(avg_latency or 0.0, 2),
                "status_breakdown": status_dist,
                "defect_breakdown": defect_dist,
            }
