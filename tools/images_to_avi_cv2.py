#!/usr/bin/env python
"""Create AVI videos from visualization PNGs using OpenCV.

This follows the project's vis_tools convention of using cv2 for image IO.
It avoids ffmpeg concat timestamp/pixel-format edge cases that can make AVI
files show as black in some desktop players.
"""

import argparse
from pathlib import Path

import cv2


DEFAULT_GROUPS = {
    "candidate_panels": "*candidate_panels.png",
    "counterfactual_case": "*counterfactual_case.png",
    "input_cameras": "*input_cameras.png",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing frame PNGs.")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument(
        "--codec",
        default="MJPG",
        help="OpenCV fourcc codec, e.g. MJPG, XVID, DIVX. Default: MJPG.",
    )
    parser.add_argument(
        "--suffix",
        default="cv2_mjpg_1280",
        help="Output filename suffix before .avi.",
    )
    return parser.parse_args()


def even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1


def make_video(frame_dir, name, pattern, fps, width, codec, suffix):
    files = sorted(frame_dir.glob(pattern))
    if not files:
        print(f"[skip] {name}: no frames matching {pattern}")
        return None

    first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {files[0]}")

    src_h, src_w = first.shape[:2]
    dst_w = even(width)
    dst_h = even(src_h * float(dst_w) / float(src_w))
    out_path = frame_dir / f"{name}_{suffix}.avi"

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (dst_w, dst_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_path} with codec={codec}")

    for idx, frame_path in enumerate(files):
        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read frame: {frame_path}")
        if img.shape[:2] != (src_h, src_w):
            raise RuntimeError(
                f"Frame size mismatch at {frame_path}: expected {(src_h, src_w)}, got {img.shape[:2]}"
            )
        resized = cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
        writer.write(resized)
        if (idx + 1) % 50 == 0 or idx + 1 == len(files):
            print(f"[{name}] wrote {idx + 1}/{len(files)} frames")

    writer.release()

    check = cv2.VideoCapture(str(out_path))
    ok, frame = check.read()
    check.release()
    if not ok or frame is None:
        raise RuntimeError(f"Video was written but cannot be decoded by OpenCV: {out_path}")

    print(f"[done] {name}: {out_path} frames={len(files)} size={dst_w}x{dst_h}")
    return out_path


def main():
    args = parse_args()
    frame_dir = Path(args.dir)
    if not frame_dir.is_dir():
        raise FileNotFoundError(frame_dir)
    if len(args.codec) != 4:
        raise ValueError("--codec must be a 4-character fourcc, e.g. MJPG")

    outputs = []
    for name, pattern in DEFAULT_GROUPS.items():
        out = make_video(frame_dir, name, pattern, args.fps, args.width, args.codec, args.suffix)
        if out is not None:
            outputs.append(out)

    print("outputs:")
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
