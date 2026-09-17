"""
VisionGuard: Automated Industrial Surface Defect & Quality Inspection System.
Developed by Tatvik Sinha.

A modular Computer Vision framework for defect localization, classification, and audit logging.
"""

__version__ = "1.0.0"
__author__ = "Tatvik Sinha"

from visionguard.config import InspectionConfig
from visionguard.preprocessor import ImagePreprocessor
from visionguard.feature_extractor import DefectFeatureExtractor
from visionguard.detector import DefectDetector, InspectionResult, DetectedDefect
from visionguard.storage import AuditStorageManager
from visionguard.dataset_generator import IndustrialSurfaceSynthesizer

__all__ = [
    "InspectionConfig",
    "ImagePreprocessor",
    "DefectFeatureExtractor",
    "DefectDetector",
    "InspectionResult",
    "DetectedDefect",
    "AuditStorageManager",
    "IndustrialSurfaceSynthesizer",
]
