"""
Procedural Industrial Surface and Defect Synthesizer.
Generates realistic textured substrates (brushed aluminum, silicon wafers, PCBs)
and mathematically synthesizes 5 defect categories with pixel-accurate ground truth masks.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from navdhrishti.config import DEFECT_CLASSES, InspectionConfig


class IndustrialSurfaceSynthesizer:
    """Generates synthetic benchmark surfaces for reproducible testing without external downloads."""

    def __init__(self, width: int = 512, height: int = 512, seed: Optional[int] = 42):
        self.width = width
        self.height = height
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def generate_base_substrate(self, substrate_type: str = "brushed_metal") -> np.ndarray:
        """
        Synthesize base background texture.
        
        Args:
            substrate_type: 'brushed_metal', 'silicon_wafer', or 'pcb_substrate'
        """
        if substrate_type == "brushed_metal":
            # Brushed metal has directional horizontal/vertical micro-streaks + Gaussian noise
            base = np.full((self.height, self.width), 170, dtype=np.float32)
            noise = np.random.normal(0, 12, (self.height, self.width)).astype(np.float32)
            # Directional blur to simulate brushing
            kernel = np.zeros((1, 15), dtype=np.float32)
            kernel[0, :] = 1.0 / 15.0
            brushed_noise = cv2.filter2D(noise, -1, kernel)
            surface = base + brushed_noise

        elif substrate_type == "silicon_wafer":
            # Ultra-smooth high reflectivity with slight radial illumination falloff
            y, x = np.ogrid[:self.height, :self.width]
            cy, cx = self.height / 2.0, self.width / 2.0
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (np.sqrt(cx**2 + cy**2))
            surface = 200.0 - 25.0 * r + np.random.normal(0, 3, (self.height, self.width))

        else:  # pcb_substrate
            # Greenish/olive substrate with grid trace patterns
            surface = np.full((self.height, self.width), 120.0, dtype=np.float32)
            # Add grid tracks
            for i in range(40, self.width, 80):
                surface[:, max(0, i - 4) : min(self.width, i + 4)] += 40.0
            for j in range(40, self.height, 80):
                surface[max(0, j - 4) : min(self.height, j + 4), :] += 30.0
            surface += np.random.normal(0, 5, (self.height, self.width))

        surface = np.clip(surface, 0, 255).astype(np.uint8)
        # Convert to 3-channel BGR
        return cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR)

    def inject_scratch(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """Inject a thin, elongated scratch defect."""
        h, w = image.shape[:2]
        x1 = random.randint(int(w * 0.15), int(w * 0.85))
        y1 = random.randint(int(h * 0.15), int(h * 0.85))
        length = random.randint(40, 140)
        angle = random.uniform(0, 2 * np.pi)

        x2 = int(np.clip(x1 + length * np.cos(angle), 10, w - 10))
        y2 = int(np.clip(y1 + length * np.sin(angle), 10, h - 10))

        thickness = random.randint(2, 4)
        # Dark scratch line with bright specular edge reflection
        cv2.line(image, (x1, y1), (x2, y2), (40, 40, 40), thickness)
        cv2.line(image, (x1 + 1, y1 + 1), (x2 + 1, y2 + 1), (240, 240, 240), 1)
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness + 1)

        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        return {
            "type": "scratch",
            "bbox": [max(0, xmin - 5), max(0, ymin - 5), (xmax - xmin) + 10, (ymax - ymin) + 10],
        }

    def inject_pinhole(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """Inject a small circular pit / pinhole defect."""
        h, w = image.shape[:2]
        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        radius = random.randint(4, 12)

        # Draw dark pit center and faint outer halo
        cv2.circle(image, (cx, cy), radius + 2, (100, 100, 100), 1)
        cv2.circle(image, (cx, cy), radius, (25, 25, 25), -1)
        cv2.circle(mask, (cx, cy), radius, 255, -1)

        return {
            "type": "pinhole",
            "bbox": [cx - radius - 2, cy - radius - 2, (radius + 2) * 2, (radius + 2) * 2],
        }

    def inject_crack(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """Inject a high-frequency zigzag micro-crack."""
        h, w = image.shape[:2]
        cx = random.randint(int(w * 0.2), int(w * 0.7))
        cy = random.randint(int(h * 0.2), int(h * 0.7))

        points = [(cx, cy)]
        curr_x, curr_y = cx, cy
        segments = random.randint(5, 9)

        for _ in range(segments):
            curr_x += random.randint(-18, 24)
            curr_y += random.randint(8, 26)
            curr_x = int(np.clip(curr_x, 5, w - 5))
            curr_y = int(np.clip(curr_y, 5, h - 5))
            points.append((curr_x, curr_y))

        pts_array = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts_array], isClosed=False, color=(20, 20, 20), thickness=2)
        cv2.polylines(mask, [pts_array], isClosed=False, color=255, thickness=4)

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return {
            "type": "crack",
            "bbox": [min(xs) - 4, min(ys) - 4, max(xs) - min(xs) + 8, max(ys) - min(ys) + 8],
        }

    def inject_contamination(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """Inject an irregular organic or chemical residue stain."""
        h, w = image.shape[:2]
        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        axes = (random.randint(18, 38), random.randint(12, 28))
        angle = random.randint(0, 180)

        overlay = image.copy()
        cv2.ellipse(overlay, (cx, cy), axes, angle, 0, 360, (50, 70, 60), -1)
        # Alpha blend to give translucent smudge appearance
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)

        rx = max(axes) + 5
        return {
            "type": "contamination",
            "bbox": [max(0, cx - rx), max(0, cy - rx), rx * 2, rx * 2],
        }

    def inject_void(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """Inject a structural material void / cavity."""
        h, w = image.shape[:2]
        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        bw = random.randint(25, 55)
        bh = random.randint(20, 45)

        cv2.rectangle(image, (cx, cy), (cx + bw, cy + bh), (15, 15, 15), -1)
        cv2.rectangle(image, (cx - 1, cy - 1), (cx + bw + 1, cy + bh + 1), (220, 220, 220), 1)
        cv2.rectangle(mask, (cx, cy), (cx + bw, cy + bh), 255, -1)

        return {
            "type": "void",
            "bbox": [cx - 2, cy - 2, bw + 4, bh + 4],
        }

    def generate_sample(
        self, defect_type: Optional[str] = None, substrate_type: str = "brushed_metal"
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Generate a complete test sample with ground-truth mask and annotations.
        
        Args:
            defect_type: One of DEFECT_CLASSES, or None for pristine (defect-free).
            substrate_type: 'brushed_metal', 'silicon_wafer', or 'pcb_substrate'
            
        Returns:
            Tuple: (synthetic_image, ground_truth_mask, metadata_list)
        """
        image = self.generate_base_substrate(substrate_type)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        defects: List[Dict] = []

        if defect_type is not None and defect_type in DEFECT_CLASSES:
            inject_funcs = {
                "scratch": self.inject_scratch,
                "pinhole": self.inject_pinhole,
                "crack": self.inject_crack,
                "contamination": self.inject_contamination,
                "void": self.inject_void,
            }
            defect_info = inject_funcs[defect_type](image, mask)
            defects.append(defect_info)

        return image, mask, defects

    def generate_dataset_batch(
        self, output_dir: Path, count_per_class: int = 4
    ) -> List[Path]:
        """
        Generate a balanced batch of pristine and defective samples with masks and metadata.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        created_paths: List[Path] = []

        # 1. Generate pristine samples
        for i in range(count_per_class):
            img, mask, _ = self.generate_sample(defect_type=None)
            img_path = output_dir / f"pristine_{i+1:02d}.png"
            mask_path = output_dir / f"pristine_{i+1:02d}_mask.png"
            cv2.imwrite(str(img_path), img)
            cv2.imwrite(str(mask_path), mask)
            created_paths.append(img_path)

        # 2. Generate defective samples for each category
        for d_class in DEFECT_CLASSES:
            for i in range(count_per_class):
                sub_type = random.choice(["brushed_metal", "silicon_wafer", "pcb_substrate"])
                img, mask, meta = self.generate_sample(defect_type=d_class, substrate_type=sub_type)
                img_path = output_dir / f"defect_{d_class}_{i+1:02d}.png"
                mask_path = output_dir / f"defect_{d_class}_{i+1:02d}_mask.png"
                meta_path = output_dir / f"defect_{d_class}_{i+1:02d}.json"

                cv2.imwrite(str(img_path), img)
                cv2.imwrite(str(mask_path), mask)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

                created_paths.append(img_path)

        return created_paths
