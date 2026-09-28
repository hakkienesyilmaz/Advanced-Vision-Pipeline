"""
Advanced Vision Pipeline - Real-Time Multi-Modal Vision & Depth Telemetry Dashboard
Main application entry point.

Usage:
    python main.py --source 0                      # Live webcam feed
    python main.py --source demo                   # Synthetic laboratory simulation
    python main.py --source data/demo_lab_video.mp4# Video file input
    python main.py --source data/lab_bench_sample.jpg # Static image
    python main.py --benchmark --max-frames 100    # Automated performance benchmark
"""

import os
import sys
import time
import argparse
from typing import Optional
import cv2
import numpy as np

from src.detector import ObjectDetector
from src.depth_estimator import DepthEstimator
from src.dashboard import DashboardCompositor
from src.telemetry import SystemTelemetry
from src.utils import VideoStreamHandler


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advanced Vision Pipeline: Real-Time Multi-Modal Vision & Depth Telemetry Dashboard"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: '0' for webcam, 'demo' for synthetic feed, or path to video/image file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-seg.pt",
        help="YOLO model checkpoint (e.g. 'yolov8n-seg.pt', 'yolov8n.pt', 'yolov8s-seg.pt')."
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default="MiDaS_small",
        help="MiDaS depth model ('MiDaS_small', 'DPT_Hybrid', 'DPT_Large')."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device for inference."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="YOLO confidence detection threshold."
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="grayscale",
        choices=["grayscale", "inferno", "magma", "jet"],
        help="Depth visualization colormap style."
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional path to save composited output video (e.g. 'output/dashboard.mp4')."
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default="",
        help="Optional path to save a single high-resolution composited snapshot image."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process (0 = infinite/until stream ends)."
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable cv2.imshow GUI display window (recommended for headless servers or automated tests)."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run in automated benchmarking mode and print latency breakdown summary."
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("  ADVANCED VISION PIPELINE - REAL-TIME SPATIAL VISION & TELEMETRY")
    print("=" * 70)

    # 1. Initialize Subsystems
    telemetry = SystemTelemetry()
    telemetry.add_log(f"[INIT] Initializing pipeline with source: '{args.source}'")

    print("[Pipeline] Loading YOLOv8 Detector...")
    detector = ObjectDetector(
        model_name=args.model,
        confidence_threshold=args.conf,
        device=args.device
    )
    telemetry.add_log(f"[INFO] Loaded detector model: {args.model}")

    print("[Pipeline] Loading MiDaS Depth Estimator...")
    depth_estimator = DepthEstimator(
        model_type=args.depth_model,
        device=args.device
    )
    telemetry.add_log(f"[INFO] Loaded depth model: {args.depth_model}")

    compositor = DashboardCompositor(target_width=1280, target_height=720)
    stream_handler = VideoStreamHandler(source=args.source)

    # 2. Setup Video Writer if requested
    video_writer: Optional[cv2.VideoWriter] = None
    if args.save:
        save_dir = os.path.dirname(args.save)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.save, fourcc, 25.0, (1280, 720))
        telemetry.add_log(f"[INFO] Video recording enabled: {args.save}")

    # Map colormap argument
    colormap_enum = None
    if args.colormap == "inferno":
        colormap_enum = cv2.COLORMAP_INFERNO
    elif args.colormap == "magma":
        colormap_enum = cv2.COLORMAP_MAGMA
    elif args.colormap == "jet":
        colormap_enum = cv2.COLORMAP_JET

    print("\n[Pipeline] System running. Press 'q' or 'ESC' in GUI to terminate.\n")
    frame_idx = 0

    benchmark_records = []

    try:
        while True:
            ret, frame = stream_handler.read_frame()
            if not ret or frame is None:
                telemetry.add_log("[WARN] Stream ended or frame read failed.")
                break

            frame_start_t = time.perf_counter()
            frame_idx += 1

            # A. YOLO Object Detection & Segmentation
            detections, yolo_lat_ms = detector.detect(frame)

            # B. MiDaS Monocular Depth Estimation
            depth_gray, depth_colored, depth_lat_ms = depth_estimator.estimate(
                frame,
                colormap=colormap_enum
            )

            # C. Spatial Sensor Fusion (Bounding Box + Depth Association)
            for det in detections:
                rel_score, depth_category = depth_estimator.extract_object_depth(
                    depth_map=depth_gray,
                    bbox=det["bbox"],
                    mask=det.get("mask")
                )
                det["depth_score"] = rel_score
                det["depth_str"] = depth_category

            # D. Render Detections Overlay
            annotated_frame = detector.draw_detections(frame, detections)

            # E. Compute Telemetry & Metrics
            pipeline_total_ms = (time.perf_counter() - frame_start_t) * 1000.0
            telemetry.update_frame_time()
            telemetry.record_latencies(yolo_lat_ms, depth_lat_ms, pipeline_total_ms)

            if args.benchmark:
                benchmark_records.append({
                    "yolo_ms": yolo_lat_ms,
                    "depth_ms": depth_lat_ms,
                    "total_ms": pipeline_total_ms
                })

            # F. Compose Unified Dashboard Canvas
            stats = telemetry.get_hardware_stats()
            logs = telemetry.get_logs()
            dashboard_canvas = compositor.compose(
                detection_frame=annotated_frame,
                depth_frame=depth_colored,
                telemetry_stats=stats,
                console_logs=logs
            )

            # G. Export / Display
            if args.snapshot and frame_idx == 3:
                snap_dir = os.path.dirname(args.snapshot)
                if snap_dir:
                    os.makedirs(snap_dir, exist_ok=True)
                cv2.imwrite(args.snapshot, dashboard_canvas)
                telemetry.add_log(f"[INFO] Snapshot captured to: {args.snapshot}")

            if video_writer is not None:
                video_writer.write(dashboard_canvas)

            if not args.no_display:
                cv2.imshow("Advanced Vision Pipeline - Real-Time Analysis Dashboard", dashboard_canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    telemetry.add_log("[INFO] User requested termination.")
                    break

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                telemetry.add_log(f"[INFO] Reached max frame limit ({args.max_frames}).")
                break

    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted by user.")
    finally:
        stream_handler.release()
        if video_writer is not None:
            video_writer.release()
            print(f"[Pipeline] Recorded video saved to: {args.save}")
        if not args.no_display:
            cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print("  SESSION SUMMARY & BENCHMARK REPORT")
    print("=" * 70)
    print(f"Total Frames Processed: {frame_idx}")
    if benchmark_records:
        yolo_vals = [b["yolo_ms"] for b in benchmark_records]
        depth_vals = [b["depth_ms"] for b in benchmark_records]
        total_vals = [b["total_ms"] for b in benchmark_records]

        print(f"Device:                  {stats.get('device', 'N/A')}")
        print(f"YOLO Inference Latency:  Mean: {np.mean(yolo_vals):.2f} ms | P95: {np.percentile(yolo_vals, 95):.2f} ms")
        print(f"MiDaS Depth Latency:     Mean: {np.mean(depth_vals):.2f} ms | P95: {np.percentile(depth_vals, 95):.2f} ms")
        print(f"Total Pipeline Latency:  Mean: {np.mean(total_vals):.2f} ms | P95: {np.percentile(total_vals, 95):.2f} ms")
        print(f"Effective Throughput:    {1000.0 / np.mean(total_vals):.2f} FPS")
    print("=" * 70)


def main():
    args = parse_arguments()
    run_pipeline(args)


if __name__ == "__main__":
    main()
