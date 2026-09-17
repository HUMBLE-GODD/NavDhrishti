"""
Command-Line Interface (CLI) for VisionGuard.
Developed by Tatvik Sinha for VITyarthi Computer Vision Evaluation.

Provides an enterprise-grade terminal UI for dataset generation, model training,
single-image inspection, batch scanning, benchmarking, and database audit logs.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import List, Optional

import cv2
import numpy as np
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from visionguard.config import DEFECT_CLASSES, InspectionConfig
from visionguard.dataset_generator import IndustrialSurfaceSynthesizer
from visionguard.detector import DefectDetector
from visionguard.storage import AuditStorageManager
from visionguard.visualizer import DefectVisualizer


console = Console()

BANNER_ART = r"""
 ██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗ 
 ██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗
 ██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║██║  ███╗██║   ██║███████║██████╔╝██║  ██║
 ╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║
  ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝
   ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ 
"""


def print_branding(title: str) -> None:
    """Render signature VisionGuard banner with Tatvik Sinha branding."""
    banner_text = Text(BANNER_ART, style="bold cyan")
    console.print(Align.center(banner_text))
    console.print(
        Align.center(
            Text.assemble(
                ("VisionGuard: Intelligent Industrial Defect Inspection System", "bold white"),
                ("  |  ", "dim"),
                ("Author: Tatvik Sinha", "bold yellow"),
                ("  |  ", "dim"),
                ("Course: Computer Vision", "bold green"),
            )
        )
    )
    console.print(Rule(title, style="cyan"))
    console.print()


def cmd_generate_samples(args: argparse.Namespace, config: InspectionConfig) -> None:
    """Generate realistic test surfaces and defects for immediate evaluation."""
    print_branding("Synthetic Surface & Defect Generator")
    output_dir = Path(args.output_dir) if args.output_dir else config.samples_dir
    count = args.count

    console.print(
        Panel(
            f"[bold cyan]Generating {count} procedural samples per defect category into:[/] [underline]{output_dir}[/]\n"
            f"[dim]Substrates: Brushed Metal, Silicon Wafer, PCB Substrate[/]\n"
            f"[dim]Defect Types: {', '.join(DEFECT_CLASSES).title()}[/]",
            title="[bold yellow]Dataset Generation Engine[/]",
            border_style="cyan",
        )
    )

    with console.status("[bold green]Synthesizing realistic surface textures and defect masks...", spinner="dots"):
        synthesizer = IndustrialSurfaceSynthesizer()
        paths = synthesizer.generate_dataset_batch(output_dir, count_per_class=count)

    console.print(f"\n[bold green]✓ Successfully synthesized {len(paths)} benchmark images with ground-truth masks![/]")
    console.print(f"Output directory: [underline]{output_dir}[/]\n")


def cmd_train(args: argparse.Namespace, config: InspectionConfig) -> None:
    """Train Random Forest classifier on procedural surface features."""
    print_branding("Machine Learning Model Training")
    console.print(
        Panel(
            f"[bold cyan]Extracting CV features and training Random Forest Classifier...[/]\n"
            f"[dim]Training samples per class: {args.samples} | Feature vector dimension: 15[/]\n"
            f"[dim]Features: Hu Moments, Canny Edge Gradients, GLCM Texture Descriptors[/]",
            title="[bold yellow]Model Optimization[/]",
            border_style="cyan",
        )
    )

    synthesizer = IndustrialSurfaceSynthesizer(seed=101)
    detector = DefectDetector(config)

    x_features: List[List[float]] = []
    y_labels: List[str] = []

    with console.status("[bold green]Extracting spatial & texture descriptors across classes...", spinner="bouncingBar"):
        for d_class in DEFECT_CLASSES:
            for _ in range(args.samples):
                img, mask, _ = synthesizer.generate_sample(defect_type=d_class)
                preprocessed = detector.preprocessor.preprocess_pipeline(img)
                _, contours = detector.segment_candidates(preprocessed)

                for c in contours:
                    feats = detector.feature_extractor.extract_contour_features(
                        c, preprocessed["gray"], preprocessed["enhanced"]
                    )
                    x_features.append(detector.feature_extractor.vector_from_dict(feats))
                    y_labels.append(d_class)

    if not x_features:
        console.print("[bold red]Error: No features extracted during training loop.[/]")
        sys.exit(1)

    detector.train_classifier(x_features, y_labels)
    console.print(
        f"\n[bold green]✓ Successfully trained classifier on {len(x_features)} extracted feature vectors![/]\n"
        f"Weights saved to: [underline]{config.model_path}[/]\n"
    )


def cmd_inspect(args: argparse.Namespace, config: InspectionConfig) -> None:
    """Inspect a single image and output color-coded terminal verdict."""
    print_branding("Single Image Inspection")
    detector = DefectDetector(config)
    visualizer = DefectVisualizer(config)
    storage = AuditStorageManager(config.db_path) if args.log_db else None

    img_path = Path(args.input)
    if not img_path.exists():
        console.print(f"[bold red]Error: Image not found:[/] {img_path}")
        sys.exit(1)

    with console.status("[bold cyan]Applying CLAHE, bilateral filter, contour segmentation & ML inference...", spinner="dots"):
        result = detector.inspect_image(img_path)

    # Log to SQLite if requested
    db_msg = ""
    if storage is not None:
        db_id = storage.log_inspection(result)
        db_msg = f"  [dim](Saved to SQLite Audit DB #{db_id})[/]"

    # Status styling
    status_style = {
        "PASS": "bold green",
        "WARNING": "bold yellow",
        "REJECT": "bold red",
    }.get(result.overall_status, "white")

    border_style = {
        "PASS": "green",
        "WARNING": "yellow",
        "REJECT": "red",
    }.get(result.overall_status, "cyan")

    console.print(
        Panel(
            f"[bold white]Target Image:[/]      [cyan]{img_path.name}[/]\n"
            f"[bold white]Inspection Result:[/] [{status_style}]{result.overall_status}[/]{db_msg}\n"
            f"[bold white]Defect Count:[/]      [bold]{result.defect_count}[/]\n"
            f"[bold white]Max Severity:[/]      [bold]{result.max_severity:.2f}[/] / 100.0\n"
            f"[bold white]Mean Severity:[/]     [bold]{result.mean_severity:.2f}[/] / 100.0\n"
            f"[bold white]Inference Latency:[/] [bold cyan]{result.inference_time_ms:.1f} ms[/] [dim](Pure CPU execution)[/]",
            title=f"Inspection Verdict: [{status_style}]{result.overall_status}[/]",
            border_style=border_style,
        )
    )

    if result.defects:
        table = Table(title="Detected Anomaly & Defect Attributes", show_header=True, header_style="bold cyan", border_style="dim")
        table.add_column("Defect ID", justify="center", style="dim")
        table.add_column("Classification", justify="left")
        table.add_column("Confidence", justify="right")
        table.add_column("Bounding Box [x, y, w, h]", justify="center")
        table.add_column("Area (px)", justify="right")
        table.add_column("Severity (0-100)", justify="right")

        for d in result.defects:
            sev_color = "red" if d.severity >= 60 else "yellow" if d.severity >= 30 else "green"
            table.add_row(
                f"#{d.defect_id}",
                f"[bold magenta]{d.defect_type.upper()}[/]",
                f"{d.confidence * 100:.1f}%",
                f"[{d.bbox[0]}, {d.bbox[1]}, {d.bbox[2]}, {d.bbox[3]}]",
                f"{d.area:.0f}",
                f"[{sev_color}]{d.severity:.1f}[/]",
            )
        console.print(table)
    else:
        console.print("[bold green]✓ Zero defects detected. Surface meets pristine quality tolerances.[/]")

    if args.save_visual:
        out_path = Path(args.output) if args.output else img_path.parent / f"{img_path.stem}_inspected.png"
        orig = cv2.imread(str(img_path))
        visualizer.save_annotated_image(orig, result, out_path)
        console.print(f"\n[bold green]✓ Annotated visual overlay saved to:[/] [underline]{out_path}[/]\n")


def cmd_batch(args: argparse.Namespace, config: InspectionConfig) -> None:
    """Run batch inspection on a folder of images with throughput reporting."""
    print_branding("Batch Quality Inspection Line")
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        console.print(f"[bold red]Error: Directory not found:[/] {input_dir}")
        sys.exit(1)

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_files = [p for p in input_dir.iterdir() if p.suffix.lower() in image_extensions and not p.name.endswith(("_mask.png", "_inspected.png", "_annotated.png"))]

    if not image_files:
        console.print(f"[yellow]No input images found in {input_dir}[/]")
        return

    detector = DefectDetector(config)
    visualizer = DefectVisualizer(config)
    storage = AuditStorageManager(config.db_path) if args.log_db else None

    console.print(
        Panel(
            f"[bold cyan]Scanning directory of {len(image_files)} manufacturing parts:[/] [underline]{input_dir}[/]\n"
            f"[dim]Logging each item to SQLite audit trail database[/]",
            title="[bold yellow]Batch Pipeline Active[/]",
            border_style="cyan",
        )
    )

    passes = 0
    warnings = 0
    rejects = 0
    total_time = 0.0

    table = Table(title="Batch Inspection Results", show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("Part Image", justify="left")
    table.add_column("Quality Verdict", justify="center")
    table.add_column("Defect Count", justify="right")
    table.add_column("Max Severity", justify="right")
    table.add_column("Latency", justify="right")

    for img_path in sorted(image_files):
        res = detector.inspect_image(img_path)
        total_time += res.inference_time_ms

        if res.overall_status == "PASS":
            passes += 1
            styled_status = "[bold green]PASS[/]"
        elif res.overall_status == "WARNING":
            warnings += 1
            styled_status = "[bold yellow]WARNING[/]"
        else:
            rejects += 1
            styled_status = "[bold red]REJECT[/]"

        table.add_row(
            img_path.name,
            styled_status,
            str(res.defect_count),
            f"{res.max_severity:.1f}",
            f"{res.inference_time_ms:.1f} ms",
        )

        if storage:
            storage.log_inspection(res)

        if args.output_dir:
            out_file = Path(args.output_dir) / f"{img_path.stem}_annotated.png"
            orig = cv2.imread(str(img_path))
            visualizer.save_annotated_image(orig, res, out_file)

    console.print(table)
    avg_latency = total_time / len(image_files)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    if storage:
        storage.log_batch_session(
            session_name=input_dir.name,
            total=len(image_files),
            passes=passes,
            warnings=warnings,
            rejects=rejects,
            mean_fps=round(fps, 1),
        )

    console.print(
        Panel(
            f"[bold green]PASS Count:[/]    {passes} ({passes / len(image_files) * 100:.1f}%)\n"
            f"[bold yellow]WARNING Count:[/] {warnings} ({warnings / len(image_files) * 100:.1f}%)\n"
            f"[bold red]REJECT Count:[/]  {rejects} ({rejects / len(image_files) * 100:.1f}%)\n"
            f"[bold cyan]Mean Latency:[/]  {avg_latency:.1f} ms per image\n"
            f"[bold cyan]Throughput:[/]    {fps:.1f} FPS (Frames Per Second on CPU)",
            title="[bold white]Production Line Summary[/]",
            border_style="green",
        )
    )


def cmd_benchmark(args: argparse.Namespace, config: InspectionConfig) -> None:
    """Evaluate detection accuracy, confusion statistics, and operational FPS."""
    print_branding("Hardware Benchmark & Evaluation Suite")
    console.print(
        Panel(
            f"[bold cyan]Executing {args.samples} trials on synthetic test substrates...[/]\n"
            f"[dim]Measuring Accuracy, Precision, Recall, F1, and Frame Rate on CPU[/]",
            title="[bold yellow]Benchmarking[/]",
            border_style="cyan",
        )
    )

    synthesizer = IndustrialSurfaceSynthesizer(seed=2026)
    detector = DefectDetector(config)

    latencies = []
    tp, fp, tn, fn = 0, 0, 0, 0

    with console.status("[bold green]Running inference trials across clean and defective surfaces...", spinner="dots"):
        # Pristine tests
        for _ in range(args.samples // 2):
            img, _, _ = synthesizer.generate_sample(defect_type=None)
            t0 = time.perf_counter()
            res = detector.inspect_image(img)
            latencies.append((time.perf_counter() - t0) * 1000.0)

            if res.overall_status == "PASS":
                tn += 1
            else:
                fp += 1

        # Defective tests
        for _ in range(args.samples // 2):
            d_class = np.random.choice(DEFECT_CLASSES)
            img, _, _ = synthesizer.generate_sample(defect_type=d_class)
            t0 = time.perf_counter()
            res = detector.inspect_image(img)
            latencies.append((time.perf_counter() - t0) * 1000.0)

            if res.overall_status in ("REJECT", "WARNING"):
                tp += 1
            else:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 1.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    avg_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    fps = 1000.0 / avg_ms

    table = Table(title="VisionGuard Laptop Performance Benchmark", show_header=True, header_style="bold green", border_style="cyan")
    table.add_column("Evaluation Metric", justify="left")
    table.add_column("Measured Performance", justify="right", style="bold")

    table.add_row("Classification Accuracy", f"{accuracy * 100:.1f}%")
    table.add_row("Defect Detection Precision", f"{precision * 100:.1f}%")
    table.add_row("Defect Detection Recall", f"{recall * 100:.1f}%")
    table.add_row("F1-Score", f"{f1:.3f}")
    table.add_row("Average CPU Latency", f"{avg_ms:.2f} ms")
    table.add_row("95th Percentile Latency (p95)", f"{p95_ms:.2f} ms")
    table.add_row("Operational Throughput", f"{fps:.1f} FPS")
    console.print(table)
    console.print()


def cmd_audit_log(args: argparse.Namespace, config: InspectionConfig) -> None:
    """Display SQLite database records and export analytics dashboard."""
    print_branding("Inspection Audit Trail & Telemetry")
    storage = AuditStorageManager(config.db_path)
    records = storage.query_recent_inspections(limit=args.limit)
    stats = storage.get_summary_statistics()

    if not records:
        console.print("[yellow]No inspection records found in the database. Run an inspection with --log-db first.[/]")
        return

    table = Table(title=f"Recent SQLite Audit Records (Showing Last {args.limit})", show_header=True, header_style="bold blue", border_style="dim")
    table.add_column("Audit ID", justify="center", style="dim")
    table.add_column("Timestamp", justify="left")
    table.add_column("Part File", justify="left")
    table.add_column("Verdict", justify="center")
    table.add_column("Defect Count", justify="right")
    table.add_column("Max Severity", justify="right")
    table.add_column("Latency", justify="right")

    for r in records:
        img_name = Path(r["image_path"]).name
        status = r["overall_status"]
        style = "bold green" if status == "PASS" else "bold yellow" if status == "WARNING" else "bold red"

        table.add_row(
            f"#{r['inspection_id']}",
            r["timestamp"],
            img_name,
            f"[{style}]{status}[/]",
            str(r["defect_count"]),
            f"{r['max_severity']:.1f}",
            f"{r['inference_time_ms']:.1f} ms",
        )
    console.print(table)

    if args.export_chart:
        visualizer = DefectVisualizer(config)
        chart_path = config.reports_dir / "audit_analytics_dashboard.png"
        visualizer.generate_analytics_chart(stats, chart_path)
        console.print(f"\n[bold green]✓ Exported telemetry analytics dashboard chart to:[/] [underline]{chart_path}[/]\n")


def main() -> None:
    """CLI Parser Configuration."""
    config = InspectionConfig()
    config.ensure_directories()

    parser = argparse.ArgumentParser(
        prog="visionguard",
        description="VisionGuard: Industrial Surface Defect & Quality Inspection CLI by Tatvik Sinha",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: generate-samples
    p_gen = subparsers.add_parser("generate-samples", help="Generate procedural test surface samples")
    p_gen.add_argument("--count", type=int, default=3, help="Number of samples to generate per category")
    p_gen.add_argument("--output-dir", type=str, default=None, help="Target directory for samples")

    # Subcommand: train
    p_train = subparsers.add_parser("train", help="Train ML classifier on procedural defect features")
    p_train.add_argument("--samples", type=int, default=35, help="Number of training samples per class")

    # Subcommand: inspect
    p_inspect = subparsers.add_parser("inspect", help="Run defect inspection on a single image")
    p_inspect.add_argument("--input", required=True, type=str, help="Path to input image file")
    p_inspect.add_argument("--save-visual", action="store_true", help="Save annotated image with bounding boxes")
    p_inspect.add_argument("--output", type=str, default=None, help="Custom output path for annotated image")
    p_inspect.add_argument("--log-db", action="store_true", default=True, help="Persist record to SQLite audit database")

    # Subcommand: batch
    p_batch = subparsers.add_parser("batch", help="Batch inspect an entire folder of images")
    p_batch.add_argument("--input-dir", required=True, type=str, help="Directory containing images")
    p_batch.add_argument("--output-dir", type=str, default=None, help="Directory to save annotated images")
    p_batch.add_argument("--log-db", action="store_true", default=True, help="Log all batch results to SQLite")

    # Subcommand: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Measure precision, recall, and FPS on laptop CPU")
    p_bench.add_argument("--samples", type=int, default=30, help="Total sample evaluations")

    # Subcommand: audit-log
    p_audit = subparsers.add_parser("audit-log", help="Query SQLite inspection history and metrics")
    p_audit.add_argument("--limit", type=int, default=10, help="Number of recent records to display")
    p_audit.add_argument("--export-chart", action="store_true", help="Generate analytics PNG plot")

    # Subcommand: generate-report
    p_rep = subparsers.add_parser("generate-report", help="Compile the official 15-section project report PDF")

    args = parser.parse_args()

    if args.command == "generate-samples":
        cmd_generate_samples(args, config)
    elif args.command == "train":
        cmd_train(args, config)
    elif args.command == "inspect":
        cmd_inspect(args, config)
    elif args.command == "batch":
        cmd_batch(args, config)
    elif args.command == "benchmark":
        cmd_benchmark(args, config)
    elif args.command == "audit-log":
        cmd_audit_log(args, config)
    elif args.command == "generate-report":
        from generate_report import build_pdf_report
        pdf_path = build_pdf_report()
        console.print(f"[bold green]✓ Successfully generated 15-section PDF report:[/] [underline]{pdf_path}[/]")
    else:
        print_branding("Command Center")
        parser.print_help()


if __name__ == "__main__":
    main()
