"""
Image Preprocessing and Illumination Normalization Module.
Applies CLAHE, bilateral filtering, and morphological background subtraction.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np

from visionguard.config import InspectionConfig


class ImagePreprocessor:
    """Preprocesses raw industrial surface images for reliable defect extraction."""

    def __init__(self, config: Optional[InspectionConfig] = None):
        self.config = config or InspectionConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size,
        )

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load an image from disk with robust path and format validation.
        
        Args:
            image_path: Absolute or relative path to image.
            
        Returns:
            np.ndarray: Loaded BGR image array.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be decoded as an image.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image from path: {path}")

        return image

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to single-channel 8-bit grayscale."""
        if len(image.shape) == 2:
            return image
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE).
        Prevents over-amplification of noise while enhancing micro-defects in shadowed areas.
        """
        if len(gray.shape) != 2:
            raise ValueError("CLAHE requires a single-channel 8-bit grayscale image.")
        return self._clahe.apply(gray)

    def denoise(self, image: np.ndarray, method: str = "bilateral") -> np.ndarray:
        """
        Denoise image while preserving sharp structural and defect edges.
        
        Args:
            image: Grayscale or BGR image.
            method: 'bilateral' for edge-preserving or 'gaussian' for fast smoothing.
        """
        if method == "bilateral":
            return cv2.bilateralFilter(
                image,
                d=self.config.bilateral_d,
                sigmaColor=self.config.bilateral_sigma_color,
                sigmaSpace=self.config.bilateral_sigma_space,
            )
        return cv2.GaussianBlur(image, self.config.gaussian_kernel_size, 0)

    def normalize_illumination(self, gray: np.ndarray, kernel_radius: int = 25) -> np.ndarray:
        """
        Estimate background illumination via large-scale morphological opening
        and subtract it to remove global lighting gradients.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_radius * 2 + 1, kernel_radius * 2 + 1)
        )
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        normalized = cv2.subtract(gray, background)
        return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

    def preprocess_pipeline(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Execute full end-to-end preprocessing pipeline.
        
        Returns:
            Dict containing:
                'original': input BGR image
                'gray': grayscale conversion
                'denoised': edge-preserved denoised grayscale
                'enhanced': CLAHE contrast-enhanced image
                'normalized': illumination-flattened image
        """
        gray = self.to_grayscale(image)
        denoised = self.denoise(gray, method="bilateral")
        enhanced = self.apply_clahe(denoised)
        normalized = self.normalize_illumination(enhanced)

        return {
            "original": image,
            "gray": gray,
            "denoised": denoised,
            "enhanced": enhanced,
            "normalized": normalized,
        }
