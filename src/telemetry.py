"""
Telemetry and performance metrics collector.
Tracks FPS, per-model latency, hardware usage (CPU/RAM/GPU), and console log queue.
"""

import time
import collections
from datetime import datetime
from typing import List, Dict, Optional
import psutil
import torch


class SystemTelemetry:
    """Monitors system hardware utilization and pipeline latency metrics."""

    def __init__(self, log_history_len: int = 6):
        self.log_history_len = log_history_len
        self.logs: collections.deque = collections.deque(maxlen=log_history_len)

        # Latency histories (rolling window of 30 frames for smooth stats)
        self.history_window = 30
        self.yolo_latencies: collections.deque = collections.deque(maxlen=self.history_window)
        self.depth_latencies: collections.deque = collections.deque(maxlen=self.history_window)
        self.total_latencies: collections.deque = collections.deque(maxlen=self.history_window)

        # FPS calculation
        self.prev_frame_time = time.time()
        self.fps = 0.0
        self.frame_count = 0

        # Initial system logs
        self.add_log(f"[SYS] Telemetry initialized at {datetime.now().strftime('%H:%M:%S')}")
        self.check_device()

    def check_device(self) -> str:
        """Determines active hardware acceleration device."""
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            self.device_str = f"CUDA (GPU: {dev_name})"
            self.add_log(f"[INFO] Hardware acceleration: CUDA enabled ({dev_name})")
        else:
            self.device_str = "CPU (Optimized SIMD)"
            self.add_log("[INFO] Hardware acceleration: CPU execution pipeline")
        return self.device_str

    def add_log(self, message: str) -> None:
        """Appends a new timestamped log message to the rolling console."""
        ts = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{ts}] {message}"
        self.logs.append(formatted)

    def update_frame_time(self) -> float:
        """Updates and returns smoothed FPS based on inter-frame duration."""
        current_time = time.time()
        dt = current_time - self.prev_frame_time
        self.prev_frame_time = current_time
        self.frame_count += 1

        instant_fps = 1.0 / dt if dt > 0 else 0.0
        # Exponential smoothing (alpha = 0.1)
        if self.fps == 0.0:
            self.fps = instant_fps
        else:
            self.fps = (0.9 * self.fps) + (0.1 * instant_fps)
        return self.fps

    def record_latencies(self, yolo_ms: float, depth_ms: float, total_ms: float) -> None:
        """Records inference latencies in milliseconds."""
        self.yolo_latencies.append(yolo_ms)
        self.depth_latencies.append(depth_ms)
        self.total_latencies.append(total_ms)

        # Periodically emit rolling statistics to console log
        if self.frame_count % 90 == 0:
            avg_tot = sum(self.total_latencies) / max(len(self.total_latencies), 1)
            self.add_log(f"[STATS] Mean Latency: {avg_tot:.1f}ms | Smoothed FPS: {self.fps:.1f}")

    def get_hardware_stats(self) -> Dict[str, str]:
        """Collects live CPU and RAM (and GPU VRAM if available) statistics."""
        cpu_pct = psutil.cpu_percent()
        ram_info = psutil.virtual_memory()
        ram_used_gb = ram_info.used / (1024 ** 3)
        ram_total_gb = ram_info.total / (1024 ** 3)

        gpu_info = "N/A"
        gpu_mem = "N/A"
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                gpu_mem = f"{allocated:.1f}GB / {reserved:.1f}GB"
                gpu_info = "Active"
            except Exception:
                gpu_info = "CUDA Error"

        return {
            "device": self.device_str,
            "cpu_percent": f"{cpu_pct:.1f}%",
            "ram_usage": f"{ram_used_gb:.1f}/{ram_total_gb:.1f} GB ({ram_info.percent:.0f}%)",
            "gpu_vram": gpu_mem,
            "fps": f"{self.fps:.1f}",
            "yolo_latency_ms": f"{sum(self.yolo_latencies)/max(len(self.yolo_latencies), 1):.1f}ms",
            "depth_latency_ms": f"{sum(self.depth_latencies)/max(len(self.depth_latencies), 1):.1f}ms",
            "total_latency_ms": f"{sum(self.total_latencies)/max(len(self.total_latencies), 1):.1f}ms",
        }

    def get_logs(self) -> List[str]:
        """Returns the recent console logs for display on the HUD console."""
        return list(self.logs)
