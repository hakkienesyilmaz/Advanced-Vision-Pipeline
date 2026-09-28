"""
Object Detection & Instance Segmentation Module using YOLOv8.
Handles model loading, inference, bounding box extraction, and segmentation masks.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import torch
from ultralytics import YOLO


class ObjectDetector:
    """Wrapper around YOLOv8 for real-time detection and segmentation."""

    def __init__(
        self,
        model_name: str = "yolov8n-seg.pt",
        confidence_threshold: float = 0.35,
        device: str = "auto"
    ):
        self.confidence_threshold = confidence_threshold

        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[ObjectDetector] Loading {model_name} on device: {self.device}")
        self.model = YOLO(model_name)
        self.model.to(self.device)

        # Warm-up inference
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False, device=self.device)
        print("[ObjectDetector] Model warmed up and ready.")

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """
        Runs object detection and instance segmentation on an input frame.

        Args:
            frame: Input image/frame in BGR format.

        Returns:
            Tuple of:
                - List of detection dictionaries (bbox, label, score, mask, center).
                - Latency in milliseconds.
        """
        t0 = time.perf_counter()

        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return detections, latency_ms

        res = results[0]
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            return detections, latency_ms

        masks = res.masks.data.cpu().numpy() if res.masks is not None else None
        h, w = frame.shape[:2]

        for i, box in enumerate(boxes):
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            label = self.model.names.get(cls_id, f"obj_{cls_id}")

            # Calculate centroid
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            det_info: Dict[str, Any] = {
                "bbox": (x1, y1, x2, y2),
                "label": label,
                "confidence": conf,
                "class_id": cls_id,
                "centroid": (cx, cy),
                "mask": None
            }

            # If segmentation mask is available, resize to frame
            if masks is not None and i < len(masks):
                raw_mask = masks[i]
                resized_mask = cv2.resize(raw_mask.astype(np.float32), (w, h)) > 0.5
                det_info["mask"] = resized_mask

            detections.append(det_info)

        return detections, latency_ms

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        color_palette: Optional[List[Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """Draws bounding boxes, labels, and alpha-blended segmentation masks on the frame."""
        output = frame.copy()
        h, w = frame.shape[:2]

        if color_palette is None:
            # High-visibility neon lab palette (BGR)
            color_palette = [
                (0, 255, 128),   # Neon Mint
                (255, 105, 180), # Neon Pink
                (0, 191, 255),   # Deep Sky Blue
                (255, 215, 0),   # Gold / Yellow
                (147, 112, 219), # Medium Purple
                (50, 205, 50),   # Lime Green
                (0, 140, 255),   # Dark Orange
            ]

        # Draw segmentation masks first
        overlay = output.copy()
        has_mask = False
        for det in detections:
            mask = det.get("mask")
            if mask is not None:
                has_mask = True
                color = color_palette[det["class_id"] % len(color_palette)]
                overlay[mask] = color

        if has_mask:
            cv2.addWeighted(overlay, 0.35, output, 0.65, 0, output)

        # Draw bounding boxes and HUD tags
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = color_palette[det["class_id"] % len(color_palette)]
            label = det["label"]
            conf = det["confidence"]
            depth_info = det.get("depth_str", "")

            # Bounding box with corner accents
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            if corner_len > 3:
                # Top-left
                cv2.line(output, (x1, y1), (x1 + corner_len, y1), color, 4)
                cv2.line(output, (x1, y1), (x1, y1 + corner_len), color, 4)
                # Bottom-right
                cv2.line(output, (x2, y2), (x2 - corner_len, y2), color, 4)
                cv2.line(output, (x2, y2), (x2, y2 - corner_len), color, 4)

            # Centroid crosshair
            cx, cy = det["centroid"]
            cv2.circle(output, (cx, cy), 3, (0, 255, 255), -1)

            # Text tag
            tag_text = f"{label} {conf:.2f}"
            if depth_info:
                tag_text += f" | {depth_info}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(tag_text, font, font_scale, thickness)

            # Label background box
            ty = max(y1 - 6, th + 6)
            cv2.rectangle(output, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), (20, 20, 25), -1)
            cv2.rectangle(output, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, 1)
            cv2.putText(output, tag_text, (x1 + 3, ty - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return output
