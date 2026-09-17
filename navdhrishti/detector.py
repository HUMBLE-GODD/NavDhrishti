"""
Defect Detection and Classification Engine.
Combines morphological contour segmentation, machine learning feature classification,
and quantitative severity scoring to render PASS/WARNING/REJECT inspection verdicts.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from navdhrishti.config import DEFECT_CLASSES, InspectionConfig
from navdhrishti.feature_extractor import DefectFeatureExtractor
from navdhrishti.preprocessor import ImagePreprocessor


@dataclass
class DetectedDefect:
    """Individual localized defect record."""
    defect_id: int
    defect_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: float
    severity: float
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class InspectionResult:
    """Complete summary of an image inspection run."""
    image_path: str
    overall_status: str  # 'PASS', 'WARNING', 'REJECT'
    defect_count: int
    max_severity: float
    mean_severity: float
    defects: List[DetectedDefect]
    inference_time_ms: float
    binary_mask: np.ndarray


class DefectDetector:
    """Core detection, localization, and classification pipeline."""

    def __init__(self, config: Optional[InspectionConfig] = None):
        self.config = config or InspectionConfig()
        self.preprocessor = ImagePreprocessor(self.config)
        self.feature_extractor = DefectFeatureExtractor()
        self.classifier: Optional[RandomForestClassifier] = None
        self._load_model_if_exists()

    def _load_model_if_exists(self) -> None:
        """Load trained classifier weights if present on disk."""
        if self.config.model_path.exists():
            try:
                self.classifier = joblib.load(str(self.config.model_path))
            except Exception:
                self.classifier = None

    def segment_candidates(self, preprocessed: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Segment candidate defect contours using adaptive gradient thresholding
        and morphological refinement.
        """
        enhanced = preprocessed["enhanced"]
        normalized = preprocessed["normalized"]

        # Canny edge detection on contrast-enhanced image
        edges = cv2.Canny(
            enhanced,
            threshold1=self.config.canny_lower,
            threshold2=self.config.canny_upper,
        )

        # Morphological gradient to capture abrupt texture changes
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.config.morph_kernel_size, self.config.morph_kernel_size),
        )
        morph_grad = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)
        _, thresh_grad = cv2.threshold(morph_grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Adaptive thresholding for low-contrast blotches/voids
        adapt_thresh = cv2.adaptiveThreshold(
            normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
        )

        # Combine edge response with gradient and threshold maps
        combined_mask = cv2.bitwise_or(edges, thresh_grad)
        combined_mask = cv2.bitwise_or(combined_mask, adapt_thresh)

        # Morphological Closing to fuse fragmented crack lines and fill pinhole interiors
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, close_kernel)

        # Morphological Opening to remove isolated single-pixel salt noise
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        clean_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, open_kernel)

        # Find connected contour boundaries
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_h, img_w = enhanced.shape[:2]
        total_img_area = img_h * img_w
        valid_contours = []

        final_mask = np.zeros_like(clean_mask)
        for c in contours:
            area = cv2.contourArea(c)
            # Filter noise and boundary artifacts
            if area < self.config.min_defect_area_px:
                continue
            if (area / total_img_area) > self.config.max_defect_area_ratio:
                continue

            x, y, w, h = cv2.boundingRect(c)
            # Filter boundary touches
            if x <= 1 or y <= 1 or (x + w) >= (img_w - 1) or (y + h) >= (img_h - 1):
                continue

            valid_contours.append(c)
            cv2.drawContours(final_mask, [c], -1, 255, -1)

        return final_mask, valid_contours

    def classify_defect(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify candidate defect using trained ML model or robust heuristic fallback.
        """
        if self.classifier is not None:
            vec = np.array([self.feature_extractor.vector_from_dict(features)])
            probs = self.classifier.predict_proba(vec)[0]
            best_idx = int(np.argmax(probs))
            conf = float(probs[best_idx])
            pred_class = str(self.classifier.classes_[best_idx])
            return pred_class, conf

        # Deterministic Heuristic Fallback (Expert Decision Tree)
        aspect_ratio = features["aspect_ratio"]
        circularity = features["circularity"]
        solidity = features["solidity"]
        mean_grad = features["mean_gradient"]
        area = features["area"]

        if aspect_ratio > 3.2 or aspect_ratio < 0.31:
            return "scratch", 0.91
        if circularity > 0.62 and area < 400:
            return "pinhole", 0.94
        if mean_grad > 35.0 and circularity < 0.35:
            return "crack", 0.88
        if solidity < 0.65 and area > 500:
            return "contamination", 0.85
        if solidity >= 0.70 and area >= 300:
            return "void", 0.89

        return "void", 0.70

    def compute_severity(self, features: Dict[str, float], img_shape: Tuple[int, int]) -> float:
        """
        Compute continuous Defect Severity Index (0.0 - 100.0).
        Factors in spatial footprint ratio, edge gradient sharpness, and texture distortion.
        """
        img_area = float(img_shape[0] * img_shape[1])
        area_ratio = features["area"] / max(img_area, 1.0)

        # Non-linear scaling: larger defects rapidly escalate severity
        area_score = min(50.0, area_ratio * 1500.0)
        grad_score = min(25.0, features["mean_gradient"] * 0.35)
        contrast_score = min(25.0, features["intensity_contrast"] * 0.25)

        total_severity = area_score + grad_score + contrast_score
        return round(float(np.clip(total_severity, 0.0, 100.0)), 2)

    def determine_status(self, defect_count: int, max_severity: float) -> str:
        """Render automated production quality status."""
        if defect_count == 0:
            return "PASS"
        if max_severity >= self.config.reject_severity_threshold or defect_count >= 3:
            return "REJECT"
        if max_severity >= self.config.warning_severity_threshold:
            return "WARNING"
        return "PASS"

    def inspect_image(self, image_input: Union[str, Path, np.ndarray]) -> InspectionResult:
        """
        Run end-to-end defect inspection on an image.
        
        Args:
            image_input: File path or raw BGR image array.
        """
        start_time = time.perf_counter()

        if isinstance(image_input, (str, Path)):
            img_path_str = str(image_input)
            image = self.preprocessor.load_image(image_input)
        else:
            img_path_str = "in-memory-array"
            image = image_input

        # 1. Preprocessing
        preprocessed = self.preprocessor.preprocess_pipeline(image)
        gray = preprocessed["gray"]

        # 2. Candidate Segmentation
        binary_mask, contours = self.segment_candidates(preprocessed)

        # 3. Feature Extraction, Classification & Severity Scoring
        detected_defects: List[DetectedDefect] = []
        for idx, contour in enumerate(contours, start=1):
            features = self.feature_extractor.extract_contour_features(contour, gray, binary_mask)
            defect_type, confidence = self.classify_defect(features)
            x, y, w, h = cv2.boundingRect(contour)
            severity = self.compute_severity(features, gray.shape)

            detected_defects.append(
                DetectedDefect(
                    defect_id=idx,
                    defect_type=defect_type,
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    area=features["area"],
                    severity=severity,
                    features=features,
                )
            )

        # 4. Metrics & Decision Rendering
        severities = [d.severity for d in detected_defects]
        max_severity = max(severities) if severities else 0.0
        mean_severity = float(np.mean(severities)) if severities else 0.0
        status = self.determine_status(len(detected_defects), max_severity)
        inference_time = (time.perf_counter() - start_time) * 1000.0

        return InspectionResult(
            image_path=img_path_str,
            overall_status=status,
            defect_count=len(detected_defects),
            max_severity=max_severity,
            mean_severity=round(mean_severity, 2),
            defects=detected_defects,
            inference_time_ms=round(inference_time, 2),
            binary_mask=binary_mask,
        )

    def train_classifier(self, x_features: List[List[float]], y_labels: List[str]) -> RandomForestClassifier:
        """
        Train and persist Random Forest classifier on extracted features.
        """
        clf = RandomForestClassifier(
            n_estimators=60,
            max_depth=8,
            random_state=42,
            class_weight="balanced",
        )
        clf.fit(x_features, y_labels)
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, str(self.config.model_path))
        self.classifier = clf
        return clf
