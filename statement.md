# Project Statement: NavDhrishti

## 1. Problem Statement
In modern precision manufacturing (e.g., semiconductor fabrication, printed circuit board [PCB] assembly, aerospace metallurgy, and automotive sheet-metal production), surface defects such as scratches, structural voids, cracks, micro-pinholes, and chemical contaminations represent major drivers of product failures, economic waste, and safety risks. 

Traditional visual quality inspection relies heavily on manual human oversight. This legacy approach suffers from several critical bottlenecks:
- **Human Fatigue & Subjectivity**: Visual fatigue sets in rapidly, leading to inconsistent defect classification and missed anomalies during long manufacturing shifts.
- **Throughput Limits**: Human inspection cannot scale to high-speed automated assembly lines operating at dozens of components per second.
- **Lack of Quantitative Traceability**: Manual checks rarely yield auditable, pixel-level statistical metrics (such as exact surface defect area, severity scoring, or real-time database logging).

**NavDhrishti** addresses this fundamental industrial challenge by providing an automated, lightweight, edge-compatible Computer Vision inspection system. It combines classical morphological feature extraction (edge gradients, texture descriptors, contour geometry) with machine learning classification to achieve real-time defect localization, severity quantification, and regulatory audit logging on standard commodity hardware without requiring expensive GPU infrastructure.

---

## 2. Scope of the Project
The scope of the NavDhrishti project encompasses the end-to-end defect inspection lifecycle for industrial flat and textured surfaces:

- **In-Scope:**
  - **Multi-Type Defect Detection**: Automated identification, localization, and classification of five critical defect categories:
    1. Scratches (linear abrasions)
    2. Pinholes / Pits (localized surface indentations)
    3. Micro-Cracks (high-frequency stress fractures)
    4. Surface Contaminations (chemical/oil residue blotches)
    5. Structural Voids (material absence/cavities)
  - **Image Preprocessing & Illumination Normalization**: CLAHE (Contrast-Limited Adaptive Histogram Equalization) and bilateral filtering to eliminate factory lighting variations and sensor noise.
  - **Feature Engineering**: Multi-scale Canny edge gradients, Hu Moment geometric invariants, and Gray-Level Co-occurrence Matrix (GLCM) texture descriptors.
  - **Quantitative Severity Scoring**: Algorithmic scoring of detected defects on a normalized 0–100 severity index to trigger automated PASS / WARNING / REJECT decisions.
  - **Relational Audit Logging**: Fully auditable SQLite storage of every inspection event, recording timestamps, defect bounding boxes, confidence ratings, and hardware throughput metrics.
  - **CLI-First Architecture**: Comprehensive, non-GUI command-line interface supporting single-image inspection, directory batch runs, performance benchmarking, and automated PDF report compilation.
  - **Evaluation & Benchmarking**: Statistical evaluation of inspection accuracy (Precision, Recall, F1-Score, Mean Intersection-over-Union [mIoU]) and operational latency (Frames Per Second [FPS]).

- **Out-of-Scope (Future Iterations):**
  - Robotic arm physical rejection hardware actuation.
  - 3D point cloud surface reconstruction via LiDAR or stereo cameras.

---

## 3. Target Users
1. **Quality Assurance (QA) & Quality Control (QC) Engineers**: Plant engineers who configure inspection tolerance thresholds, monitor defect trends, and maintain ISO 9001 quality audit trails.
2. **Manufacturing Line Operators**: Technicians who run batch quality checks during production runs and require immediate, color-coded PASS/REJECT verdicts at the workstation terminal.
3. **Plant Automation & Systems Integrators**: Engineers seeking a lightweight, dependency-minimal computer vision library that integrates directly into embedded manufacturing execution systems (MES) without requiring specialized GPU hardware.
4. **Academic & Industrial Researchers**: Practitioners studying hybrid computer vision architectures that unite classical morphology with statistical pattern recognition.

---

## 4. High-Level Features
- **Adaptive Lighting Correction**: Built-in CLAHE and Gaussian/Bilateral filtering pipelines to handle non-uniform factory illumination and specular reflection.
- **Dual-Engine Defect Localization**: Heuristic contour-based segmentation combined with a machine learning classification engine for robust defect categorization.
- **Continuous Severity Index**: Computes a continuous 0–100 defect severity score factoring in spatial defect area ratio, gradient intensity, and texture distortion.
- **Relational Inspection Audit Trail**: Native SQLite database logging every inspection event, defect bounding box, confidence score, and processing latency.
- **Procedural Surface & Defect Synthesizer**: Generates mathematically grounded pristine and defective surfaces with pixel-accurate ground-truth masks for instant testing and reproducibility.
- **CLI Automation Suite**: Terminal interface featuring `inspect`, `batch`, `train`, `benchmark`, `audit-log`, and `generate-report` commands.
- **Automated 15-Section PDF Report Compiler**: Bundled generator that produces an official academic project report with embedded architecture and UML diagrams.
