"""
Unit tests for DefectFeatureExtractor module.
"""

import cv2
import numpy as np
import pytest

from navdhrishti.feature_extractor import DefectFeatureExtractor


@pytest.fixture
def feature_extractor() -> DefectFeatureExtractor:
    return DefectFeatureExtractor()


def test_compute_glcm_features(feature_extractor: DefectFeatureExtractor) -> None:
    # Uniform patch -> low contrast, high homogeneity
    uniform_patch = np.full((32, 32), 128, dtype=np.uint8)
    glcm_uni = feature_extractor.compute_glcm_features(uniform_patch)
    assert glcm_uni["contrast"] == pytest.approx(0.0, abs=1e-3)
    assert glcm_uni["homogeneity"] == pytest.approx(1.0, abs=1e-3)

    # High contrast checkerboard patch -> high contrast
    checker = np.indices((32, 32)).sum(axis=0) % 2 * 255
    checker = checker.astype(np.uint8)
    glcm_check = feature_extractor.compute_glcm_features(checker)
    assert glcm_check["contrast"] > 0.0


def test_extract_contour_features(feature_extractor: DefectFeatureExtractor) -> None:
    # Synthesize circular defect
    img = np.full((128, 128), 180, dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(img, (64, 64), 15, 30, -1)
    cv2.circle(mask, (64, 64), 15, 255, -1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1

    feats = feature_extractor.extract_contour_features(contours[0], img, mask)

    assert feats["circularity"] > 0.70
    assert feats["area"] > 500
    assert feats["aspect_ratio"] == pytest.approx(1.0, abs=0.2)
    assert "hu_moment_1" in feats
    assert "glcm_contrast" in feats

    vec = feature_extractor.vector_from_dict(feats)
    assert len(vec) == len(DefectFeatureExtractor.FEATURE_NAMES)
    assert all(isinstance(v, (int, float)) for v in vec)
