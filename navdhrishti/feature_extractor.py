"""
Defect Feature Extraction Module.
Computes geometric invariants (Hu moments, circularity, solidity),
Canny edge gradients, and GLCM (Gray-Level Co-occurrence Matrix) texture metrics.
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np


class DefectFeatureExtractor:
    """Extracts discriminative morphological, textural, and edge features from candidate defects."""

    FEATURE_NAMES = [
        "area",
        "perimeter",
        "circularity",
        "aspect_ratio",
        "extent",
        "solidity",
        "hu_moment_1",
        "hu_moment_2",
        "hu_moment_3",
        "glcm_contrast",
        "glcm_dissimilarity",
        "glcm_homogeneity",
        "glcm_energy",
        "mean_gradient",
        "intensity_contrast",
    ]

    def compute_glcm_features(self, patch: np.ndarray) -> Dict[str, float]:
        """
        Compute Gray-Level Co-occurrence Matrix (GLCM) statistical texture descriptors.
        Uses 16 quantized levels for high speed and robustness on laptop CPUs.
        """
        if patch.size == 0 or patch.shape[0] < 3 or patch.shape[1] < 3:
            return {
                "contrast": 0.0,
                "dissimilarity": 0.0,
                "homogeneity": 1.0,
                "energy": 1.0,
            }

        # Quantize to 16 gray levels
        quantized = (patch.astype(np.float32) / 256.0 * 16).astype(np.int32)
        h, w = quantized.shape

        # Horizontal adjacent pair matrix (distance=1, angle=0)
        glcm = np.zeros((16, 16), dtype=np.float32)
        for r in range(h):
            row = quantized[r, :]
            for c in range(w - 1):
                i = row[c]
                j = row[c + 1]
                glcm[i, j] += 1.0
                glcm[j, i] += 1.0  # Symmetric

        total_pairs = glcm.sum()
        if total_pairs > 0:
            glcm /= total_pairs
        else:
            return {
                "contrast": 0.0,
                "dissimilarity": 0.0,
                "homogeneity": 1.0,
                "energy": 1.0,
            }

        i_indices, j_indices = np.indices((16, 16))
        diff = np.abs(i_indices - j_indices)

        contrast = float(np.sum((diff**2) * glcm))
        dissimilarity = float(np.sum(diff * glcm))
        homogeneity = float(np.sum(glcm / (1.0 + diff**2)))
        energy = float(np.sum(glcm**2))

        return {
            "contrast": contrast,
            "dissimilarity": dissimilarity,
            "homogeneity": homogeneity,
            "energy": energy,
        }

    def extract_contour_features(
        self, contour: np.ndarray, gray_img: np.ndarray, binary_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract geometric and photometric feature vector for a specific defect contour.
        """
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, closed=True))

        # Circularity (4 * pi * Area / Perimeter^2)
        circularity = (4.0 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0

        # Bounding box & Aspect Ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / float(h) if h > 0 else 1.0
        extent = area / float(w * h) if (w * h) > 0 else 0.0

        # Convex Hull and Solidity
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = (area / hull_area) if hull_area > 0 else 0.0

        # Hu Moments (Log scale)
        moments = cv2.moments(contour)
        hu = cv2.HuMoments(moments).flatten()
        log_hu = []
        for val in hu[:3]:
            # Log transform with sign preservation
            log_hu.append(-1.0 * np.copysign(1.0, val) * np.log10(abs(val) + 1e-12))

        # Texture inside bounding patch
        patch = gray_img[y : y + h, x : x + w]
        glcm_feats = self.compute_glcm_features(patch)

        # Gradient magnitude using Sobel
        if patch.size > 0:
            sobelx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            mean_grad = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))
        else:
            mean_grad = 0.0

        # Intensity contrast (Defect region vs surrounding background)
        c_mask = np.zeros(gray_img.shape, dtype=np.uint8)
        cv2.drawContours(c_mask, [contour], -1, 255, -1)
        defect_pixels = gray_img[c_mask == 255]
        mean_defect = float(np.mean(defect_pixels)) if defect_pixels.size > 0 else 0.0
        mean_bg = float(np.mean(gray_img[c_mask == 0])) if (gray_img.size - defect_pixels.size) > 0 else 0.0
        intensity_contrast = abs(mean_defect - mean_bg)

        return {
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
            "extent": extent,
            "solidity": solidity,
            "hu_moment_1": float(log_hu[0]),
            "hu_moment_2": float(log_hu[1]),
            "hu_moment_3": float(log_hu[2]),
            "glcm_contrast": glcm_feats["contrast"],
            "glcm_dissimilarity": glcm_feats["dissimilarity"],
            "glcm_homogeneity": glcm_feats["homogeneity"],
            "glcm_energy": glcm_feats["energy"],
            "mean_gradient": mean_grad,
            "intensity_contrast": intensity_contrast,
        }

    def vector_from_dict(self, feats: Dict[str, float]) -> List[float]:
        """Convert feature dictionary into a strict fixed-order numeric array."""
        return [feats.get(name, 0.0) for name in self.FEATURE_NAMES]
