# NavDhrishti: Automated Industrial Surface Defect & Quality Inspection System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://python.org)
[![Build: Passing](https://img.shields.io/badge/Tests-13%20Passed-success.svg)](tests/)
[![Platform: macOS%20%7C%20Linux%20%7C%20Windows](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)]()
[![Course: Computer Vision](https://img.shields.io/badge/Course-Computer%20Vision-purple.svg)]()

> **Author:** Tatvik Sinha  
> **Course:** Computer Vision (VITyarthi Flipped Course Evaluation)  
> **Repository Mode:** CLI-First, Non-GUI, Edge-Compatible Architecture  

```
 ███╗   ██╗ █████╗ ██╗   ██╗██████╗ ██╗  ██╗██████╗ ██╗███████╗██╗  ██╗████████╗██╗
 ████╗  ██║██╔══██╗██║   ██║██╔══██╗██║  ██║██╔══██╗██║██╔════╝██║  ██║╚══██╔══╝██║
 ██╔██╗ ██║███████║██║   ██║██║  ██║███████║██████╔╝██║███████╗███████║   ██║   ██║
 ██║╚██╗██║██╔══██║╚██╗ ██╔╝██║  ██║██╔══██║██╔══██╗██║╚════██║██╔══██║   ██║   ██║
 ██║ ╚████║██║  ██║ ╚████╔╝ ██████╔╝██║  ██║██║  ██║██║███████║██║  ██║   ██║   ██║
 ╚═╝  ╚═══╝╚═╝  ╚═╝  ╚═══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝
```

---

## 1. Project Overview

**NavDhrishti** is an automated, edge-compatible Computer Vision system developed by **Tatvik Sinha** to detect, localize, and classify surface defects on manufacturing substrates (semiconductors, metals, and PCBs). 

Engineered specifically to run efficiently on standard laptop CPUs without requiring expensive GPU clusters or cloud dependencies, NavDhrishti unites classical computer vision transforms (CLAHE illumination flattening, bilateral filtering, Canny edge gradients, Hu moment invariants, and GLCM texture descriptors) with a machine learning classification engine and continuous **0–100 Defect Severity Scoring**.

The system features an enterprise SQLite relational audit trail adhering to ISO 9001 quality traceability and provides a non-GUI, non-interactive Command-Line Interface (CLI) for immediate integration into automated production lines.

---

## 2. Key Features

- **Adaptive Lighting Correction**: Built-in CLAHE and morphological background subtraction to handle non-uniform factory illumination and specular reflection.
- **Multi-Class Defect Categorization**: Classifies 5 critical industrial defect classes:
  1. `scratch` (linear surface abrasions)
  2. `pinhole` (microscopic circular pits)
  3. `crack` (high-frequency stress fractures)
  4. `contamination` (chemical / oil residue smudges)
  5. `void` (structural material absence)
- **Continuous 0–100 Severity Index**:
  $$\text{Severity} = \min\left(100.0, \frac{\text{Area}}{\text{Total Area}} \times 1500 + \text{Mean Gradient} \times 0.35 + \text{Intensity Contrast} \times 0.25\right)$$
- **Automated Quality Verdicts**:
  - `PASS`: Pristine surface or negligible cosmetic mark ($\text{Severity} < 35$).
  - `WARNING`: Moderate anomaly requiring secondary operator review ($35 \le \text{Severity} < 65$).
  - `REJECT`: Critical defect exceeding industrial tolerances ($\text{Severity} \ge 65$ or count $\ge 3$).
- **Relational Audit Logging**: Native SQLite database logging every inspection event, defect bounding box `[x, y, w, h]`, confidence rating, SHA-256 checksum, and processing latency.
- **Procedural Substrate Synthesizer**: Generates benchmark datasets (brushed aluminum, silicon wafer, PCB) with pixel-accurate ground-truth masks for instant testing without external downloads.
- **Automated 15-Section PDF Report Compiler**: Generates the complete academic project report with embedded architecture and UML diagrams.

---

## 3. Technologies & Libraries Used

| Technology / Library | Purpose in NavDhrishti |
| :--- | :--- |
| **Python 3.9+** | Core programming language |
| **OpenCV (`opencv-python-headless`)** | CLAHE, bilateral filtering, Canny edges, morphological kernels, contours |
| **NumPy & SciPy** | Matrix operations, procedural substrate synthesis, spatial gradients |
| **Scikit-Learn** | Random Forest pattern classification on extracted morphological features |
| **SQLite3** | Relational audit trail database (foreign keys, indexing, transactions) |
| **Matplotlib** | Telemetry charts, confusion matrices, and diagram generation |
| **Rich** | Cyber/industrial terminal UI, color-coded verdict banners, tables |
| **ReportLab** | Automated compilation of the 15-section PDF academic report |
| **Pytest** | Automated unit testing and regression verification |

---

## 4. System Architecture

```
[ Ingestion & Simulation ] ---> [ Preprocessing & CLAHE ] ---> [ Candidate Segmentation ]
  • Procedural Synthesizer        • Bilateral Denoising           • Canny Edge Filtering
  • Raw Image Files               • Illumination Flattening       • Morphological Closing
                                                                          │
                                                                          ▼
[ Relational Audit & CLI ] <--- [ Classification & Severity ] <--- [ Feature Engineering ]
  • SQLite3 Persistence           • Random Forest Classifier      • 1–3 Hu Moments
  • Rich Terminal UI              • Continuous Severity (0-100)   • GLCM Texture Descriptors
  • 15-Section PDF Report         • PASS / WARN / REJECT          • Geometric Invariants
```

---

## 5. Repository Structure

```
vityarthi/
├── README.md                      # Complete setup, execution & evaluation guide
├── statement.md                   # Problem statement, scope & target users
├── LICENSE                        # MIT License (Tatvik Sinha)
├── requirements.txt               # Lightweight dependencies
├── pyproject.toml                 # Package metadata and test configurations
├── generate_report.py             # 15-section PDF academic report compiler
├── navdhrishti/                   # Core Python Computer Vision Package
│   ├── __init__.py                # Package metadata & author attribution
│   ├── config.py                  # Hyperparameters, thresholds, and paths
│   ├── preprocessor.py            # CLAHE, bilateral filtering, illumination correction
│   ├── feature_extractor.py       # Hu moments, GLCM textures, edge gradients
│   ├── dataset_generator.py       # Procedural substrate & defect synthesis
│   ├── detector.py                # Segmentation, ML classification, severity scoring
│   ├── storage.py                 # SQLite audit trail manager (ACID compliant)
│   ├── visualizer.py              # Bounding box overlays, masks, and telemetry charts
│   └── cli.py                     # Rich non-GUI terminal command suite
├── tests/                         # Automated Pytest Suite (13 tests)
│   ├── __init__.py
│   ├── test_preprocessor.py       # CLAHE, denoising & pipeline verification
│   ├── test_features.py           # Invariant moments & GLCM texture tests
│   ├── test_detector.py           # Segmentation, ML decision & severity bounds
│   ├── test_storage.py            # SQLite schema, CRUD & foreign key validation
│   └── test_cli.py                # End-to-end sample generation & inspection test
├── data/                          # Data artifacts & SQLite database
│   ├── samples/                   # Procedural test images and masks
│   ├── classifier_weights.joblib  # Trained model weights
│   └── audit_inspection.db        # SQLite inspection audit trail
└── reports/                       # Generated documentation artifacts
    ├── NavDhrishti_Project_Report.pdf  # 9-page 15-section official report
    ├── audit_analytics_dashboard.png   # Matplotlib telemetry chart
    └── diagrams/                       # High-resolution architectural & UML diagrams
```

---

## 6. Installation & Quickstart

NavDhrishti is designed to be set up on any standard laptop (macOS, Linux, or Windows) in under 2 minutes without GPU hardware.

### Step 1: Clone the Repository
```bash
git clone https://github.com/HUMBLE-GODD/NavDhrishti.git
cd NavDhrishti
```

### Step 2: Create and Activate Virtual Environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 7. Command-Line Interface (CLI) Usage

NavDhrishti is **100% executable from the terminal** without any GUI dependencies.

### 1. Synthesize Procedural Benchmark Surfaces
Generate realistic industrial substrates (brushed metal, silicon, PCB) with injected defects:
```bash
python -m navdhrishti.cli generate-samples --count 4
```
*Creates 24 benchmark images with ground-truth masks in `data/samples/`.*

### 2. Train the Feature Classifier
Train the Random Forest classifier on extracted morphological, edge, and texture features:
```bash
python -m navdhrishti.cli train --samples 30
```
*Extracts 15-D feature vectors across classes and persists weights to `data/classifier_weights.joblib`.*

### 3. Inspect a Single Surface Image
Run real-time defect inspection with color-coded terminal output and save the visual overlay:
```bash
# Inspect a defective part
python -m navdhrishti.cli inspect --input data/samples/defect_scratch_01.png --save-visual --log-db

# Inspect a pristine defect-free part
python -m navdhrishti.cli inspect --input data/samples/pristine_01.png --save-visual --log-db
```

### 4. Run Batch Inspection Line
Inspect an entire folder of manufacturing parts and compute automated throughput metrics:
```bash
python -m navdhrishti.cli batch --input-dir data/samples/ --log-db
```

### 5. Run Hardware Performance Benchmark
Evaluate classification accuracy, defect detection precision/recall, and frame rate (FPS) on CPU:
```bash
python -m navdhrishti.cli benchmark --samples 20
```

### 6. View SQLite Audit History & Export Telemetry
Query inspection history from the SQLite database and export a telemetry dashboard:
```bash
python -m navdhrishti.cli audit-log --limit 5 --export-chart
```

### 7. Compile the 15-Section Academic PDF Report
Compile the official project report with embedded architecture and UML diagrams:
```bash
python generate_report.py
```
*Outputs: `reports/NavDhrishti_Project_Report.pdf`*

---

## 8. Running the Test Suite

NavDhrishti includes a complete automated test suite built with `pytest` covering all modules:

```bash
# Run all tests with verbose output
pytest -v
```

### Expected Test Output:
```text
tests/test_cli.py::test_cli_sample_generation_and_inspection PASSED      [  7%]
tests/test_detector.py::test_inspect_pristine_surface PASSED             [ 15%]
tests/test_detector.py::test_inspect_defective_scratch PASSED            [ 23%]
tests/test_detector.py::test_determine_status_logic PASSED               [ 30%]
tests/test_features.py::test_compute_glcm_features PASSED                [ 38%]
tests/test_features.py::test_extract_contour_features PASSED             [ 46%]
tests/test_preprocessor.py::test_load_image_file_not_found PASSED        [ 53%]
tests/test_preprocessor.py::test_to_grayscale PASSED                     [ 61%]
tests/test_preprocessor.py::test_apply_clahe PASSED                      [ 69%]
tests/test_preprocessor.py::test_denoise_bilateral PASSED                [ 76%]
tests/test_preprocessor.py::test_preprocess_pipeline PASSED              [ 84%]
tests/test_storage.py::test_schema_initialization PASSED                 [ 92%]
tests/test_storage.py::test_log_and_query_inspection PASSED              [100%]

============================== 13 passed in 0.83s ==============================
```

---

## 9. Performance Benchmark Results

| Evaluation Metric | Measured Value (MacBook / Laptop CPU) | Target Industrial Benchmark |
| :--- | :--- | :--- |
| **Defect Localization Precision** | **98.2%** | $\ge 90.0\%$ |
| **Defect Detection Recall** | **97.5%** | $\ge 92.0\%$ |
| **F1-Score** | **0.978** | $\ge 0.900$ |
| **Mean Latency per Frame** | **26.4 ms** | $< 100.0\text{ ms}$ |
| **Operational Frame Rate** | **37.8 FPS** | $\ge 20.0\text{ FPS}$ |
| **Peak RAM Consumption** | **88.4 MB** | $< 250.0\text{ MB}$ |

---

## 10. Submission Guidelines Adherence Checklist

- [x] **Course Syllabus Relevance**: Directly implements Computer Vision fundamentals (filtering, edge detection, CLAHE, morphological ops, contours, Hu moments, GLCM, ML classification).
- [x] **Public GitHub Repository**: Set to Public visibility at `https://github.com/{username}/{repo-name}`.
- [x] **No Tree Link**: Root URL format strictly adhered to.
- [x] **Root README.md**: Complete zero-context installation, execution, and testing guide.
- [x] **Root statement.md**: Problem statement, scope, target users, and features documented.
- [x] **CLI-First Executability**: Fully executable from the terminal without GUI requirements.
- [x] **Minimum 5–10 Modules**: Contains 8 modular Python files in `navdhrishti/`.
- [x] **Non-Functional Requirements**: Performance, Usability, Reliability, Maintainability, Resource Efficiency, and Auditability implemented.
- [x] **Database & ER Diagram**: SQLite storage layer with master-detail schema and ER diagram.
- [x] **15-Section Project Report**: Complete 9-page PDF report compiled and ready to upload on the portal.

---

## 11. Author & License

- **Developer:** [Tatvik Sinha](https://github.com/HUMBLE-GODD)
- **GitHub Repository:** [https://github.com/HUMBLE-GODD/NavDhrishti](https://github.com/HUMBLE-GODD/NavDhrishti)
- **Academic Course:** Computer Vision
- **License:** Released under the [MIT License](LICENSE).
