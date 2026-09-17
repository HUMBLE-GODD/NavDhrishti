"""
NavDhrishti: Automated Industrial Surface Defect & Quality Inspection System.
Developed by Tatvik Sinha.

A modular Computer Vision framework for defect localization, classification, and audit logging.
"""

__version__ = "1.0.0"
__author__ = "Tatvik Sinha"

from navdhrishti.config import InspectionConfig
from navdhrishti.preprocessor import ImagePreprocessor
from navdhrishti.feature_extractor import DefectFeatureExtractor
from navdhrishti.detector import DefectDetector, InspectionResult, DetectedDefect
from navdhrishti.storage import AuditStorageManager
from navdhrishti.dataset_generator import IndustrialSurfaceSynthesizer

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
