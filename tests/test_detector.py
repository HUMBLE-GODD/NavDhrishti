"""
Unit tests for DefectDetector and decision engine.
"""

import numpy as np
import pytest

from navdhrishti.config import InspectionConfig
from navdhrishti.dataset_generator import IndustrialSurfaceSynthesizer
from navdhrishti.detector import DefectDetector


@pytest.fixture
def detector() -> DefectDetector:
    config = InspectionConfig()
    return DefectDetector(config)


@pytest.fixture
def synthesizer() -> IndustrialSurfaceSynthesizer:
    return IndustrialSurfaceSynthesizer(seed=77)


def test_inspect_pristine_surface(detector: DefectDetector, synthesizer: IndustrialSurfaceSynthesizer) -> None:
    img, _, _ = synthesizer.generate_sample(defect_type=None)
    result = detector.inspect_image(img)

    assert result.overall_status == "PASS"
    assert result.defect_count == 0
    assert result.max_severity == 0.0
    assert result.inference_time_ms > 0.0


def test_inspect_defective_scratch(detector: DefectDetector, synthesizer: IndustrialSurfaceSynthesizer) -> None:
    img, _, _ = synthesizer.generate_sample(defect_type="scratch")
    result = detector.inspect_image(img)

    assert result.defect_count >= 1
    assert result.max_severity > 0.0
    assert result.overall_status in ("WARNING", "REJECT")
    assert len(result.defects) == result.defect_count
    for d in result.defects:
        x, y, w, h = d.bbox
        assert x >= 0 and y >= 0
        assert w > 0 and h > 0
        assert 0.0 <= d.severity <= 100.0


def test_determine_status_logic(detector: DefectDetector) -> None:
    assert detector.determine_status(0, 0.0) == "PASS"
    assert detector.determine_status(1, 20.0) == "PASS"
    assert detector.determine_status(1, 45.0) == "WARNING"
    assert detector.determine_status(1, 80.0) == "REJECT"
    assert detector.determine_status(4, 30.0) == "REJECT"
