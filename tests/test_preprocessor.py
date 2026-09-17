"""
Unit tests for ImagePreprocessor module.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from visionguard.config import InspectionConfig
from visionguard.preprocessor import ImagePreprocessor


@pytest.fixture
def preprocessor() -> ImagePreprocessor:
    return ImagePreprocessor(InspectionConfig())


@pytest.fixture
def sample_bgr_image() -> np.ndarray:
    """Create a synthetic 200x200 3-channel test image."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :] = (120, 130, 140)
    cv2.circle(img, (100, 100), 20, (30, 30, 30), -1)
    return img


def test_load_image_file_not_found(preprocessor: ImagePreprocessor) -> None:
    with pytest.raises(FileNotFoundError):
        preprocessor.load_image("non_existent_image_path_12345.png")


def test_to_grayscale(preprocessor: ImagePreprocessor, sample_bgr_image: np.ndarray) -> None:
    gray = preprocessor.to_grayscale(sample_bgr_image)
    assert gray.ndim == 2
    assert gray.shape == (200, 200)
    assert gray.dtype == np.uint8


def test_apply_clahe(preprocessor: ImagePreprocessor, sample_bgr_image: np.ndarray) -> None:
    gray = preprocessor.to_grayscale(sample_bgr_image)
    enhanced = preprocessor.apply_clahe(gray)
    assert enhanced.shape == gray.shape
    assert enhanced.dtype == np.uint8
    # CLAHE should boost contrast of the inner circle
    assert enhanced.std() >= gray.std()


def test_denoise_bilateral(preprocessor: ImagePreprocessor, sample_bgr_image: np.ndarray) -> None:
    gray = preprocessor.to_grayscale(sample_bgr_image)
    denoised = preprocessor.denoise(gray, method="bilateral")
    assert denoised.shape == gray.shape
    assert denoised.dtype == np.uint8


def test_preprocess_pipeline(preprocessor: ImagePreprocessor, sample_bgr_image: np.ndarray) -> None:
    out = preprocessor.preprocess_pipeline(sample_bgr_image)
    assert "original" in out
    assert "gray" in out
    assert "denoised" in out
    assert "enhanced" in out
    assert "normalized" in out
    assert out["normalized"].shape == (200, 200)
