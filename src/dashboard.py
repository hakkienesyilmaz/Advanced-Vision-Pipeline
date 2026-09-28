"""
Dashboard Compositor for Antigravity AI Vision System.
Combines Object Detection, MiDaS Depth Estimation, and Technical Console into a unified 3-panel HUD.
"""

from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import cv2


class DashboardCompositor:
    """Assembles detection stream, depth map, and telemetry HUD into a single canvas."""

    def __init__(self, target_width: int = 1280, target_height: int = 720):
        self.canvas_w = target_width
        self.canvas_h = target_height

        # Panel coordinates and layout (1280x720 standard HD)
        self.header_h = 38
        self.footer_h = 175
        self.pad = 8

        # Panel dimensions
        self.top_h = self.canvas_h - self.header_h - self.footer_h - (3 * self.pad)
        self.panel_w = (self.canvas_w - (3 * self.pad)) // 2

        # Left panel (Detection):
        self.left_x1 = self.pad
        self.left_x2 = self.left_x1 + self.panel_w
        self.top_y1 = self.header_h + self.pad
        self.top_y2 = self.top_y1 + self.top_h

        # Right panel (Depth):
        self.right_x1 = self.left_x2 + self.pad
        self.right_x2 = self.right_x1 + self.panel_w

        # Bottom panel (Telemetry & Console):
        self.bot_x1 = self.pad
        self.bot_x2 = self.canvas_w - self.pad
        self.bot_y1 = self.top_y2 + self.pad
        self.bot_y2 = self.canvas_h - self.pad

        # Theme Colors (BGR)
        self.bg_color = (14, 16, 20)           # Deep Dark Slate
        self.panel_bg = (20, 24, 30)           # Panel Background
        self.border_color = (45, 55, 68)       # Subtle Border
        self.accent_cyan = (235, 206, 0)       # Cyan / Electric Blue (BGR)
        self.accent_green = (120, 220, 80)     # Neon Green
        self.accent_red = (80, 80, 240)        # Status Alert Red
        self.text_dim = (160, 165, 175)        # Dim Gray
        self.text_bright = (245, 245, 245)     # Bright White

    def compose(
        self,
        detection_frame: np.ndarray,
        depth_frame: np.ndarray,
        telemetry_stats: Dict[str, str],
        console_logs: List[str]
    ) -> np.ndarray:
        """
        Builds the unified 3-panel dashboard frame.

        Args:
            detection_frame: Frame containing YOLO annotations.
            depth_frame: Frame containing MiDaS depth visualization.
            telemetry_stats: Dictionary of current FPS, latency, and hardware stats.
            console_logs: List of recent log messages.

        Returns:
            Rendered canvas as a BGR numpy array.
        """
        # Create base canvas
        canvas = np.full((self.canvas_h, self.canvas_w, 3), self.bg_color, dtype=np.uint8)

        # 1. Render Top Header
        self._render_header(canvas, telemetry_stats)

        # 2. Render Left Panel (Object Detection & Segmentation Feed)
        self._render_feed_panel(
            canvas,
            frame=detection_frame,
            x1=self.left_x1, y1=self.top_y1,
            x2=self.left_x2, y2=self.top_y2,
            title="STEREO CAMERA FEED: AI-DRIVEN OBJECT ANALYSIS",
            subtitle=f"Inference: {telemetry_stats.get('yolo_latency_ms', '14.0ms')} (Optimized)"
        )

        # 3. Render Right Panel (Monocular Depth Map)
        self._render_depth_panel(
            canvas,
            frame=depth_frame,
            x1=self.right_x1, y1=self.top_y1,
            x2=self.right_x2, y2=self.top_y2,
            title="MiDaS RELATIVE DEPTH MAP",
            subtitle=f"Inference: {telemetry_stats.get('depth_latency_ms', '24.0ms')}"
        )

        # 4. Render Bottom Panel (Technical Console & Telemetry Metrics)
        self._render_bottom_panel(
            canvas,
            x1=self.bot_x1, y1=self.bot_y1,
            x2=self.bot_x2, y2=self.bot_y2,
            stats=telemetry_stats,
            logs=console_logs
        )

        return canvas

    def _render_header(self, canvas: np.ndarray, stats: Dict[str, str]) -> None:
        """Draws top title bar, logo, and active device indicator."""
        # Top banner background
        cv2.rectangle(canvas, (0, 0), (self.canvas_w, self.header_h), (18, 22, 28), -1)
        cv2.line(canvas, (0, self.header_h), (self.canvas_w, self.header_h), (50, 60, 75), 1)

        # Glowing Antigravity AI logo icon
        cv2.circle(canvas, (24, self.header_h // 2), 7, self.accent_cyan, -1)
        cv2.circle(canvas, (24, self.header_h // 2), 3, (255, 255, 255), -1)

        # Title
        cv2.putText(
            canvas,
            "Antigravity AI",
            (40, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Subtitle badge
        cv2.putText(
            canvas,
            "LABORATORY TELEMETRY SYSTEM v2.6",
            (190, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            self.accent_cyan,
            1,
            cv2.LINE_AA
        )

        # Right side: Timestamp & Device badge
        now_str = datetime.now().strftime("%b %d, %Y | %H:%M:%S UTC")
        dev_str = stats.get("device", "CPU")
        right_text = f"{dev_str}  |  {now_str}"

        (tw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(
            canvas,
            right_text,
            (self.canvas_w - tw - 16, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self.text_dim,
            1,
            cv2.LINE_AA
        )

    def _render_feed_panel(
        self,
        canvas: np.ndarray,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        title: str,
        subtitle: str
    ) -> None:
        """Renders video feed with border, header, and metadata overlay."""
        # Panel container box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.panel_bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.border_color, 1)

        # Title bar
        title_h = 24
        cv2.rectangle(canvas, (x1, y1), (x2, y1 + title_h), (25, 30, 38), -1)
        cv2.putText(
            canvas,
            title,
            (x1 + 10, y1 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (230, 230, 230),
            1,
            cv2.LINE_AA
        )

        # Resize feed to fit inner panel
        inner_x1 = x1 + 2
        inner_y1 = y1 + title_h + 1
        inner_w = (x2 - x1) - 4
        inner_h = (y2 - inner_y1) - 2

        if inner_w > 10 and inner_h > 10:
            resized = cv2.resize(frame, (inner_w, inner_h))
            canvas[inner_y1:inner_y1 + inner_h, inner_x1:inner_x1 + inner_w] = resized

        # Bottom info overlay on video
        overlay_y = y2 - 6
        cv2.putText(
            canvas,
            subtitle,
            (x1 + 8, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 200),
            1,
            cv2.LINE_AA
        )

    def _render_depth_panel(
        self,
        canvas: np.ndarray,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        title: str,
        subtitle: str
    ) -> None:
        """Renders depth map with depth scale indicator (Near <-> Far)."""
        self._render_feed_panel(canvas, frame, x1, y1, x2, y2, title, subtitle)

        # Draw Depth Scale indicator (Near/Far) on top-right of depth feed
        scale_x = x2 - 80
        scale_y = y1 + 35
        cv2.putText(canvas, "[DEPTH SCALE]", (scale_x - 10, scale_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.accent_cyan, 1)
        cv2.putText(canvas, "^ Near (White)", (scale_x - 10, scale_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1)
        cv2.putText(canvas, "v Far (Dark)", (scale_x - 10, scale_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.30, self.text_dim, 1)

    def _render_bottom_panel(
        self,
        canvas: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        stats: Dict[str, str],
        logs: List[str]
    ) -> None:
        """Renders the lower split panel: console logs on left, telemetry cards on right."""
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (12, 14, 18), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.border_color, 1)

        # Panel header
        header_h = 22
        cv2.rectangle(canvas, (x1, y1), (x2, y1 + header_h), (22, 26, 34), -1)
        cv2.putText(
            canvas,
            "TECHNICAL CONSOLE & METRICS",
            (x1 + 10, y1 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self.accent_cyan,
            1,
            cv2.LINE_AA
        )

        # Split width: 65% for console logs, 35% for telemetry metrics cards
        split_x = x1 + int((x2 - x1) * 0.65)
        cv2.line(canvas, (split_x, y1), (split_x, y2), self.border_color, 1)

        # --- LEFT SUB-PANEL: Python Console Logs ---
        log_start_y = y1 + header_h + 18
        line_height = 20
        max_lines = (y2 - log_start_y) // line_height

        recent_logs = logs[-max_lines:] if len(logs) > max_lines else logs
        for i, log_line in enumerate(recent_logs):
            ly = log_start_y + (i * line_height)
            color = self.accent_green if "[STATS]" in log_line else self.text_dim
            if "[SYS]" in log_line or "[INFO]" in log_line:
                color = (200, 230, 255)

            cv2.putText(
                canvas,
                log_line,
                (x1 + 14, ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                color,
                1,
                cv2.LINE_AA
            )

        # Blinking cursor simulation
        cursor_y = log_start_y + (len(recent_logs) * line_height)
        if cursor_y < y2 - 5:
            cv2.putText(
                canvas,
                "> _",
                (x1 + 14, cursor_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                self.accent_cyan,
                1,
                cv2.LINE_AA
            )

        # --- RIGHT SUB-PANEL: Telemetry Telemetry Cards ---
        card_start_x = split_x + 15
        card_y = y1 + header_h + 16

        items = [
            ("Throughput (FPS)", stats.get("fps", "0.0"), self.accent_green),
            ("YOLO Latency", stats.get("yolo_latency_ms", "0.0ms"), (0, 220, 255)),
            ("MiDaS Latency", stats.get("depth_latency_ms", "0.0ms"), (255, 180, 0)),
            ("Total Pipeline", stats.get("total_latency_ms", "0.0ms"), self.accent_cyan),
            ("System RAM", stats.get("ram_usage", "N/A"), self.text_dim),
            ("CPU Load", stats.get("cpu_percent", "0%"), self.text_dim),
        ]

        # Draw 2 columns of 3 metrics
        col_w = (x2 - card_start_x) // 2
        for idx, (label, val, color) in enumerate(items):
            col = idx % 2
            row = idx // 2
            cx = card_start_x + (col * col_w)
            cy = card_y + (row * 36)

            cv2.putText(canvas, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.text_dim, 1, cv2.LINE_AA)
            cv2.putText(canvas, val, (cx, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
