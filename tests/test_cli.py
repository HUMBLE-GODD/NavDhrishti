"""
Integration and CLI command tests for NavDhrishti.
"""

from pathlib import Path
import pytest

from navdhrishti.config import InspectionConfig
from navdhrishti.dataset_generator import IndustrialSurfaceSynthesizer
from navdhrishti.detector import DefectDetector
from navdhrishti.storage import AuditStorageManager


def test_cli_sample_generation_and_inspection(tmp_path: Path) -> None:
    config = InspectionConfig()
    config.samples_dir = tmp_path / "samples"
    config.db_path = tmp_path / "test_cli.db"
    config.ensure_directories()

    # Generate samples
    synthesizer = IndustrialSurfaceSynthesizer(seed=12)
    sample_paths = synthesizer.generate_dataset_batch(config.samples_dir, count_per_class=1)
    assert len(sample_paths) > 0
    assert sample_paths[0].exists()

    # Inspect sample
    detector = DefectDetector(config)
    result = detector.inspect_image(sample_paths[0])
    assert result.overall_status in ("PASS", "WARNING", "REJECT")

    # Persist to test storage
    storage = AuditStorageManager(config.db_path)
    insp_id = storage.log_inspection(result)
    assert insp_id == 1

    records = storage.query_recent_inspections(limit=5)
    assert len(records) == 1
