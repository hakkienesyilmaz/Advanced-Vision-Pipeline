"""
Utility functions for video stream handling, frame generation, and export.
"""

import os
import time
from typing import Generator, Tuple, Optional
import numpy as np
import cv2


class VideoStreamHandler:
    """Manages video frames from webcams, files, images, or synthetic generator."""

    def __init__(self, source: str = "0", loop: bool = True):
        self.source = source
        self.loop = loop
        self.cap = None
        self.is_synthetic = False
        self.is_image = False
        self.cached_image = None
        self._init_source()

    def _init_source(self) -> None:
        if self.source.lower() == "demo" or self.source.lower() == "synthetic":
            self.is_synthetic = True
            print("[VideoStreamHandler] Using synthetic laboratory simulation feed.")
            return

        # Check if source is a numeric string (webcam index)
        if self.source.isdigit():
            cam_idx = int(self.source)
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            if not self.cap.isOpened():
                print(f"[VideoStreamHandler] Warning: Webcam {cam_idx} not found. Falling back to synthetic demo feed.")
                self.is_synthetic = True
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return

        # Check if image file
        ext = os.path.splitext(self.source)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            if os.path.exists(self.source):
                self.is_image = True
                self.cached_image = cv2.imread(self.source)
                print(f"[VideoStreamHandler] Loaded still image: {self.source}")
                return
            else:
                print(f"[VideoStreamHandler] Image {self.source} not found, falling back to synthetic feed.")
                self.is_synthetic = True
                return

        # Otherwise treat as video file
        if os.path.exists(self.source):
            self.cap = cv2.VideoCapture(self.source)
        else:
            print(f"[VideoStreamHandler] Video source '{self.source}' not found. Falling back to synthetic feed.")
            self.is_synthetic = True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads the next available frame."""
        if self.is_synthetic:
            return True, self._generate_synthetic_frame()

        if self.is_image:
            time.sleep(0.03)  # Emulate ~30fps
            return True, self.cached_image.copy()

        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret and self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            return ret, frame

        return False, None

    def _generate_synthetic_frame(self) -> np.ndarray:
        """
        Synthesizes a realistic laboratory workbench scene with dynamic moving elements
        (calibration target, power supply, robotic arm, sensor core prototype).
        Allows testing without webcam or video files.
        """
        h, w = 480, 640
        t = time.time()

        # Lab workbench background gradient
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Background wall
        img[:220, :] = (45, 45, 52)
        # Workbench surface
        img[220:, :] = (70, 75, 85)
        # Horizon line
        cv2.line(img, (0, 220), (w, 220), (30, 30, 35), 2)

        # 1. Background Rack & Instruments
        cv2.rectangle(img, (40, 60), (220, 210), (35, 38, 44), -1)
        cv2.rectangle(img, (40, 60), (220, 210), (60, 65, 75), 2)
        for y in range(80, 200, 25):
            cv2.line(img, (50, y), (210, y), (80, 90, 105), 1)

        # 2. Calibration Target (concentric circles)
        cx, cy = 380, 160
        cv2.rectangle(img, (320, 100), (440, 220), (220, 220, 220), -1)
        for r in [50, 38, 26, 14, 4]:
            color = (0, 160, 190) if (r // 12) % 2 == 0 else (240, 240, 240)
            cv2.circle(img, (cx, cy), r, color, -1)
            cv2.circle(img, (cx, cy), r, (40, 40, 40), 1)

        # Crosshair on target
        cv2.line(img, (cx - 55, cy), (cx + 55, cy), (0, 0, 0), 1)
        cv2.line(img, (cx, cy - 55), (cx, cy + 55), (0, 0, 0), 1)

        # 3. Power Supply Unit (Yellow Bench Box)
        cv2.rectangle(img, (120, 260), (260, 380), (40, 170, 200), -1)
        cv2.rectangle(img, (120, 260), (260, 380), (30, 120, 150), 2)
        # Digital Display
        cv2.rectangle(img, (140, 280), (200, 310), (10, 10, 15), -1)
        cv2.putText(img, "24.0V", (145, 302), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        # Dials
        cv2.circle(img, (230, 295), 10, (30, 30, 35), -1)
        cv2.circle(img, (230, 340), 10, (30, 30, 35), -1)

        # 4. Optical Sensor Core Prototype (Glows with sinusoidal pulse)
        pulse = 0.5 + 0.5 * np.sin(t * 3.0)
        core_x = int(450 + 20 * np.sin(t * 0.8))
        core_y = 330
        radius = 55

        # Base stand
        cv2.rectangle(img, (core_x - 45, core_y + radius - 15), (core_x + 45, core_y + radius + 15), (35, 35, 40), -1)
        # Outer ring
        cv2.circle(img, (core_x, core_y), radius, (50, 55, 65), -1)
        cv2.circle(img, (core_x, core_y), radius, (100, 110, 130), 3)

        # Glowing reactor coils
        glow_val = int(180 + 75 * pulse)
        cv2.circle(img, (core_x, core_y), radius - 12, (glow_val, int(glow_val * 0.8), 20), 4)
        cv2.circle(img, (core_x, core_y), radius - 24, (255, 200, 50), 2)
        cv2.circle(img, (core_x, core_y), 12, (255, 255, 255), -1)

        # 5. Robotic Arm Articulation (Animated joint)
        arm_base_x, arm_base_y = 70, 420
        joint1_x = 70 + int(30 * np.cos(t * 1.2))
        joint1_y = 300 + int(15 * np.sin(t * 1.2))
        gripper_x = joint1_x + 60 + int(20 * np.sin(t * 1.2))
        gripper_y = joint1_y - 40 + int(20 * np.cos(t * 1.2))

        # Draw arm segments
        cv2.line(img, (arm_base_x, arm_base_y), (joint1_x, joint1_y), (140, 40, 180), 16)
        cv2.line(img, (joint1_x, joint1_y), (gripper_x, gripper_y), (180, 50, 220), 12)
        cv2.circle(img, (joint1_x, joint1_y), 10, (220, 220, 220), -1)
        cv2.circle(img, (arm_base_x, arm_base_y), 16, (100, 30, 140), -1)

        # Gripper fingers
        cv2.line(img, (gripper_x, gripper_y), (gripper_x + 15, gripper_y - 10), (200, 200, 200), 3)
        cv2.line(img, (gripper_x, gripper_y), (gripper_x + 15, gripper_y + 10), (200, 200, 200), 3)

        return img

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
