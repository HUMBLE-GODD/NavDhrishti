"""
Configuration settings and constants for NavDhrishti inspection pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# Defect Category Constants
DEFECT_CLASSES = [
    "scratch",
    "pinhole",
    "crack",
    "contamination",
    "void"
]

DEFECT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "scratch": (0, 165, 255),       # Orange (BGR)
    "pinhole": (0, 255, 255),       # Yellow (BGR)
    "crack": (0, 0, 255),           # Red (BGR)
    "contamination": (255, 0, 255), # Magenta (BGR)
    "void": (255, 0, 0),            # Blue (BGR)
}


@dataclass
class InspectionConfig:
    """Centralized hyperparameters and system thresholds."""
    # Preprocessing
    clahe_clip_limit: float = 2.5
    clahe_grid_size: Tuple[int, int] = (8, 8)
    bilateral_d: int = 7
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    gaussian_kernel_size: Tuple[int, int] = (5, 5)

    # Edge and Contour Detection
    canny_lower: int = 40
    canny_upper: int = 130
    morph_kernel_size: int = 3
    min_defect_area_px: int = 25
    max_defect_area_ratio: float = 0.40

    # Decision Boundaries (0 - 100 severity scale)
    warning_severity_threshold: float = 35.0
    reject_severity_threshold: float = 65.0

    # Storage and paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    db_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "audit_inspection.db")
    samples_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "samples")
    model_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "classifier_weights.joblib")
    reports_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "reports")

    def ensure_directories(self) -> None:
        """Ensure all runtime directories exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
