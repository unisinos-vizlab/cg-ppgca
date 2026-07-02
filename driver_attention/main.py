"""
Driver Attention Monitor — entry point.

Usage:
    python -m driver_attention.main                # webcam 0
    python -m driver_attention.main --source 1     # webcam 1
    python -m driver_attention.main --source video.mp4

Keyboard controls:
    q / ESC  — quit
    r        — reset monitor (awareness + calibration)
    c        — clear calibration only
    SPACE    — pause / resume
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2

from .face_analyzer import FaceAnalyzer
from .monitor import DriverAttentionMonitor
from .display import AttentionDisplay, FaceData


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Driver Attention Monitor (OpenPilot)")
    p.add_argument("--source", default="0",
                   help="Video source: camera index (int) or file path")
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--save",   default="",
                   help="Optional path to save output video (e.g. output.mp4)")
    return p.parse_args()


def open_capture(source: str, width: int, height: int) -> cv2.VideoCapture:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}", file=sys.stderr)
        sys.exit(1)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def run(args: argparse.Namespace) -> None:
    cap     = open_capture(args.source, args.width, args.height)
    analyzer = FaceAnalyzer()
    monitor  = DriverAttentionMonitor()
    display  = AttentionDisplay()

    writer: cv2.VideoWriter | None = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 20, (1024, 560))

    # Empty face/state for first frame
    dummy_face  = FaceData()
    from .monitor import AttentionState
    dummy_state = AttentionState(
        awareness=1.0, alert_level=0, face_detected=False,
        is_distracted=False, distracted_types={},
        calibrated=False, calib_percent=0,
        pitch=0.0, yaw=0.0, blink_left=0.0, blink_right=0.0,
        terminal_count=0, distraction_filter=0.0,
    )

    paused   = False
    prev_t   = time.monotonic()
    fps_smooth = 30.0

    print("[INFO] Driver Attention Monitor started.")
    print("[INFO] Controls: q=quit  r=reset  c=clear-calib  SPACE=pause")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # q or ESC
            break
        if key == ord('r'):
            monitor.reset()
            print("[INFO] Monitor reset.")
        if key == ord('c'):
            monitor._calib_samples = 0
            print("[INFO] Calibration cleared.")
        if key == ord(' '):
            paused = not paused
            print("[INFO]", "Paused." if paused else "Resumed.")

        if paused:
            # Re-render last frame while paused
            canvas = display.render(last_frame, dummy_face, dummy_state, fps_smooth)
            cv2.putText(canvas, "PAUSADO", (440, 270),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 200, 255), 3, cv2.LINE_AA)
            cv2.imshow("Driver Attention Monitor", canvas)
            continue

        ret, frame = cap.read()
        if not ret:
            # Loop video files
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        last_frame = frame.copy()

        now = time.monotonic()
        dt  = now - prev_t
        dt  = max(1e-3, min(dt, 0.2))   # clamp to [1ms, 200ms]
        prev_t = now
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)

        # Analyse face
        face  = analyzer.analyze(frame)
        # Update attention model
        state = monitor.update(face, dt)

        # Render dashboard
        canvas = display.render(frame, face, state, fps_smooth)

        cv2.imshow("Driver Attention Monitor", canvas)
        if writer is not None:
            writer.write(canvas)

        dummy_face  = face
        dummy_state = state

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
