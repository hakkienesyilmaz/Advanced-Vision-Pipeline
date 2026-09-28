"""
Monocular Depth Estimation Module using MiDaS.
Produces relative depth maps and provides depth extraction for detected objects.
"""

import time
from typing import Tuple, Optional
import numpy as np
import cv2
import torch


class DepthEstimator:
    """Monocular Depth Estimator using PyTorch Hub MiDaS."""

    def __init__(self, model_type: str = "MiDaS_small", device: str = "auto"):
        self.model_type = model_type

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[DepthEstimator] Loading {model_type} on {self.device}...")

        # Load MiDaS model and transforms from torch hub (trusted)
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        print(f"[DepthEstimator] {model_type} initialized and ready.")

    def estimate(
        self,
        frame: np.ndarray,
        colormap: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Infers relative depth from BGR frame.

        Args:
            frame: Input BGR image (H, W, 3).
            colormap: Optional OpenCV colormap (e.g., cv2.COLORMAP_INFERNO).

        Returns:
            Tuple of:
                - Grayscale normalized depth map (uint8, 0-255: closer is brighter).
                - Colored depth visualization (BGR, uint8).
                - Latency in milliseconds.
        """
        t0 = time.perf_counter()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply MiDaS preprocessing transform
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            # Resize depth prediction back to original frame dimensions
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_np = prediction.cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Normalize to 0-255 uint8 (higher value = closer to camera in MiDaS)
        depth_min = depth_np.min()
        depth_max = depth_np.max()

        if depth_max - depth_min > 1e-6:
            depth_normalized = ((depth_np - depth_min) / (depth_max - depth_min) * 255.0).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_np, dtype=np.uint8)

        # Create depth visualization
        if colormap is not None:
            depth_colored = cv2.applyColorMap(depth_normalized, colormap)
        else:
            # Grayscale 3-channel visualization
            depth_colored = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)

        return depth_normalized, depth_colored, latency_ms

    @staticmethod
    def extract_object_depth(
        depth_map: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mask: Optional[np.ndarray] = None
    ) -> Tuple[float, str]:
        """
        Computes median depth value within the detected object region.

        Args:
            depth_map: Grayscale normalized depth map (0-255).
            bbox: (x1, y1, x2, y2) bounds.
            mask: Optional binary segmentation mask.

        Returns:
            Tuple of (relative_score [0.0-1.0], category_str).
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0, "Unknown"

        if mask is not None and np.any(mask[y1:y2, x1:x2]):
            region_depths = depth_map[y1:y2, x1:x2][mask[y1:y2, x1:x2]]
        else:
            region_depths = depth_map[y1:y2, x1:x2]

        if len(region_depths) == 0:
            return 0.0, "Unknown"

        # Robust median depth
        median_val = float(np.median(region_depths))
        rel_score = median_val / 255.0  # 1.0 = closest, 0.0 = furthest

        # Categorize spatial proximity
        if rel_score > 0.65:
            cat = f"Near ({rel_score:.2f})"
        elif rel_score > 0.35:
            cat = f"Mid ({rel_score:.2f})"
        else:
            cat = f"Far ({rel_score:.2f})"

        return rel_score, cat
