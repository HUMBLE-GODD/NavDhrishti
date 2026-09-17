"""
Visualization and Annotation Module.
Draws color-coded bounding boxes, severity badges, and generates diagnostic charts.
"""

from pathlib import Path
from typing import Optional, Union
import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless terminal execution
import matplotlib.pyplot as plt
import numpy as np

from navdhrishti.config import DEFECT_COLORS, InspectionConfig
from navdhrishti.detector import InspectionResult


class DefectVisualizer:
    """Renders visual inspection overlays, annotations, and statistical diagnostic charts."""

    def __init__(self, config: Optional[InspectionConfig] = None):
        self.config = config or InspectionConfig()

    def render_overlay(
        self, original_image: np.ndarray, result: InspectionResult
    ) -> np.ndarray:
        """
        Draw bounding boxes, defect labels, semi-transparent mask highlights,
        and header status banner onto the image.
        """
        annotated = original_image.copy()
        h, w = annotated.shape[:2]

        # 1. Overlay defect masks with alpha blending
        if result.binary_mask is not None and np.any(result.binary_mask > 0):
            mask_colored = np.zeros_like(annotated)
            mask_colored[result.binary_mask > 0] = (0, 0, 255)  # Red mask
            cv2.addWeighted(mask_colored, 0.35, annotated, 0.65, 0, annotated)

        # 2. Draw bounding boxes and labels for each defect
        for defect in result.defects:
            x, y, bw, bh = defect.bbox
            color = DEFECT_COLORS.get(defect.defect_type, (0, 255, 0))

            # Bounding rectangle
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

            # Label text
            label = f"{defect.defect_type.upper()} ({int(defect.confidence * 100)}%) Sev:{defect.severity}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            # Label background banner
            label_y = max(y - 5, text_h + 5)
            cv2.rectangle(
                annotated,
                (x, label_y - text_h - 4),
                (x + text_w + 6, label_y + baseline),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x + 3, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # 3. Header Status Banner
        status_colors = {
            "PASS": (40, 180, 40),      # Green
            "WARNING": (0, 165, 255),   # Amber
            "REJECT": (30, 30, 220),    # Red
        }
        banner_color = status_colors.get(result.overall_status, (100, 100, 100))
        cv2.rectangle(annotated, (0, 0), (w, 36), banner_color, -1)

        banner_text = (
            f"STATUS: {result.overall_status} | DEFECTS: {result.defect_count} | "
            f"MAX SEV: {result.max_severity} | LATENCY: {result.inference_time_ms:.1f}ms"
        )
        cv2.putText(
            annotated,
            banner_text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return annotated

    def save_annotated_image(
        self, original_image: np.ndarray, result: InspectionResult, output_path: Union[str, Path]
    ) -> Path:
        """Render and persist annotated inspection image to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        annotated = self.render_overlay(original_image, result)
        cv2.imwrite(str(path), annotated)
        return path

    def generate_analytics_chart(self, stats: dict, output_path: Union[str, Path]) -> Path:
        """
        Generate multi-panel matplotlib summary dashboard covering:
        1. Defect Category Distribution
        2. Inspection Status Breakdown (Pass / Warning / Reject)
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
        fig.suptitle("NavDhrishti Quality Analytics & Telemetry", fontsize=14, fontweight="bold")

        # Panel 1: Defect Distribution
        defects = stats.get("defect_breakdown", {})
        if defects:
            classes = list(defects.keys())
            counts = list(defects.values())
            bars = ax1.bar(classes, counts, color="#3498db", edgecolor="#2980b9", width=0.55)
            ax1.set_title("Defect Classification Distribution", fontsize=11)
            ax1.set_ylabel("Count")
            ax1.grid(axis="y", linestyle="--", alpha=0.6)
            for bar in bars:
                h = bar.get_height()
                ax1.annotate(
                    f"{h}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        else:
            ax1.text(0.5, 0.5, "No Defect Records Yet", ha="center", va="center")
            ax1.set_title("Defect Classification Distribution")

        # Panel 2: Status Breakdown
        statuses = stats.get("status_breakdown", {})
        status_palette = {"PASS": "#2ecc71", "WARNING": "#f39c12", "REJECT": "#e74c3c"}
        if statuses:
            labels = list(statuses.keys())
            values = list(statuses.values())
            colors = [status_palette.get(lbl, "#95a5a6") for lbl in labels]
            ax2.pie(
                values,
                labels=labels,
                autopct="%1.1f%%",
                startangle=140,
                colors=colors,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            )
            ax2.set_title("Overall Inspection Status Ratio", fontsize=11)
        else:
            ax2.text(0.5, 0.5, "No Inspection Records Yet", ha="center", va="center")
            ax2.set_title("Overall Inspection Status Ratio")

        plt.tight_layout()
        plt.savefig(str(path))
        plt.close(fig)
        return path
