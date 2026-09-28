"""
Utility script to generate offline sample data:
- lab_bench_sample.jpg (test still image)
- demo_lab_video.mp4 (10-second test video clip)
"""

import os
import sys
import numpy as np
import cv2

# Add parent directory to sys.path to import src.utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import VideoStreamHandler


def generate_samples(output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    stream = VideoStreamHandler(source="demo")

    # 1. Generate still test image
    image_path = os.path.join(output_dir, "lab_bench_sample.jpg")
    _, sample_frame = stream.read_frame()
    if sample_frame is not None:
        cv2.imwrite(image_path, sample_frame)
        print(f"[SampleGenerator] Generated test image: {image_path}")

    # 2. Generate 10-second demo video (300 frames @ 30 FPS)
    video_path = os.path.join(output_dir, "demo_lab_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = sample_frame.shape[:2]
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))

    print(f"[SampleGenerator] Rendering 300 frames into {video_path}...")
    for _ in range(300):
        _, frame = stream.read_frame()
        out.write(frame)

    out.release()
    print(f"[SampleGenerator] Demo video successfully generated: {video_path}")


if __name__ == "__main__":
    generate_samples()
