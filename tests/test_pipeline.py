"""
Unit and Integration Tests for Advanced Vision Pipeline.
Tests ObjectDetector, DepthEstimator, Sensor Fusion, and Dashboard Compositor.
"""

import os
import sys
import unittest
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detector import ObjectDetector
from src.depth_estimator import DepthEstimator
from src.dashboard import DashboardCompositor
from src.telemetry import SystemTelemetry
from src.utils import VideoStreamHandler


class TestVisionPipeline(unittest.TestCase):
    """Test suite ensuring all pipeline subsystems run smoothly without failure."""

    @classmethod
    def setUpClass(cls):
        # Create a test dummy frame (480x640 BGR)
        cls.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a synthetic bright circle and rectangle to simulate objects
        cv2.circle(cls.dummy_frame, (320, 240), 60, (200, 200, 200), -1)
        cv2.rectangle(cls.dummy_frame, (100, 100), (220, 300), (0, 180, 220), -1)

        cls.detector = ObjectDetector(model_name="yolov8n-seg.pt", device="cpu")
        cls.depth_estimator = DepthEstimator(model_type="MiDaS_small", device="cpu")
        cls.compositor = DashboardCompositor(target_width=1280, target_height=720)
        cls.telemetry = SystemTelemetry()

    def test_detector_inference(self):
        """Test YOLOv8 detector returns valid structure and non-negative latency."""
        detections, latency_ms = self.detector.detect(self.dummy_frame)
        self.assertIsInstance(detections, list)
        self.assertGreaterEqual(latency_ms, 0.0)

    def test_depth_estimation(self):
        """Test MiDaS depth estimator produces valid dimensions and depth map."""
        depth_gray, depth_colored, latency_ms = self.depth_estimator.estimate(self.dummy_frame)
        self.assertEqual(depth_gray.shape, (480, 640))
        self.assertEqual(depth_colored.shape, (480, 640, 3))
        self.assertEqual(depth_gray.dtype, np.uint8)
        self.assertGreater(latency_ms, 0.0)

    def test_depth_extraction(self):
        """Test spatial depth extraction from bounding box region."""
        depth_map = np.full((480, 640), 200, dtype=np.uint8)
        score, category = self.depth_estimator.extract_object_depth(
            depth_map=depth_map,
            bbox=(50, 50, 200, 200)
        )
        self.assertAlmostEqual(score, 200 / 255.0, places=2)
        self.assertIn("Near", category)

    def test_dashboard_composition(self):
        """Test HUD compositor produces an exact 1280x720 canvas."""
        annotated = self.dummy_frame.copy()
        _, depth_colored, _ = self.depth_estimator.estimate(self.dummy_frame)
        stats = self.telemetry.get_hardware_stats()
        logs = self.telemetry.get_logs()

        canvas = self.compositor.compose(
            detection_frame=annotated,
            depth_frame=depth_colored,
            telemetry_stats=stats,
            console_logs=logs
        )
        self.assertEqual(canvas.shape, (720, 1280, 3))
        self.assertEqual(canvas.dtype, np.uint8)

    def test_synthetic_stream_handler(self):
        """Test fallback synthetic stream generator produces frames."""
        handler = VideoStreamHandler(source="demo")
        ret, frame = handler.read_frame()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        handler.release()


if __name__ == "__main__":
    unittest.main()
