import argparse
import logging
from pathlib import Path
from datetime import datetime
import csv
import time

import cv2
import numpy as np

from .logging_config import setup_logging
from .head_pose_estimator import (
    estimate_head_pose,
    estimate_head_pose_heuristic,
    estimate_head_pose_mediapipe,
    estimate_head_pose_hybrid,
    draw_pose_overlay,
    HeadPoseError,
    euler_from_rvec,
    HeadPoseResult,
)


def _classify_gaze(pitch_deg: float, yaw_deg: float) -> str:
    """Roughly classify gaze zone from pitch/yaw angles."""
    if abs(yaw_deg) < 10.0 and abs(pitch_deg) < 10.0:
        return "CENTER"
    if yaw_deg <= -10.0 and abs(pitch_deg) < 25.0:
        return "LEFT"
    if yaw_deg >= 10.0 and abs(pitch_deg) < 25.0:
        return "RIGHT"
    if pitch_deg <= -10.0:
        return "UP"
    if pitch_deg >= 10.0:
        return "DOWN"
    return "UNKNOWN"


def _quality_from_error(err):
    """Convert reprojection error (pixels) into a simple quality label."""
    if err is None:
        return "n/a"
    if err < 3.0:
        return "good"
    if err < 7.0:
        return "ok"
    return "poor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gaze-estimate",
        description=(
            "Estimate head pose (pitch, yaw, roll) from an image or webcam. "
            "Uses MediaPipe + solvePnP when available, with OpenCV fallback."
        ),
    )
    parser.add_argument(
        "image",
        type=str,
        nargs="?",
        help="Path to input image (PNG/JPEG) when not using webcam",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to save an overlay image (image mode only)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["auto", "mediapipe", "heuristic", "hybrid"],
        default="auto",
        help=(
            "Pose estimation method. 'hybrid' uses MediaPipe when reprojection "
            "error is low, otherwise falls back to heuristic."
        ),
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Run in webcam/continuous mode",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--stabilize",
        type=float,
        default=0.0,
        help="Apply EMA smoothing to pose in webcam mode (alpha 0..1; 0=off)",
    )
    parser.add_argument(
        "--record-csv",
        type=str,
        default=None,
        help=(
            "Optional path to a CSV file where pose, gaze zone, and quality "
            "will be logged (single image or webcam)."
        ),
    )
    parser.add_argument(
        "--attention-summary",
        action="store_true",
        help=(
            "In webcam mode, print a dwell-time summary per gaze zone at the "
            "end of the session."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def _estimate_single_image(
    image_path: Path,
    method: str,
    logger: logging.Logger,
    record_csv: str | None = None,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error("Failed to read image: %s", image_path)
        raise SystemExit(1)

    try:
        if method == "heuristic":
            result = estimate_head_pose_heuristic(image)
        elif method == "mediapipe":
            result = estimate_head_pose_mediapipe(image)
        elif method == "hybrid":
            result = estimate_head_pose_hybrid(image)
        else:
            result = estimate_head_pose(image)
    except HeadPoseError as exc:
        logger.error("Head pose estimation failed: %s", exc)
        raise SystemExit(1)
    except Exception:
        logger.exception("Unexpected error during head pose estimation")
        raise SystemExit(1)

    print(f"Pitch: {result.pitch_deg:.2f} degrees")
    print(f"Yaw:   {result.yaw_deg:.2f} degrees")
    print(f"Roll:  {result.roll_deg:.2f} degrees")

    zone = _classify_gaze(result.pitch_deg, result.yaw_deg)
    print(f"Gaze zone: {zone}")
    reproj_err = getattr(result, "reprojection_error", None)
    if reproj_err is not None:
        quality = _quality_from_error(reproj_err)
        print(
            f"Pose quality: {quality} (reprojection error {reproj_err:.2f} px; lower is better)"
        )

    if record_csv is not None:
        csv_path = Path(record_csv)
        file_exists = csv_path.is_file()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "timestamp",
                        "mode",
                        "image_path",
                        "method",
                        "pitch_deg",
                        "yaw_deg",
                        "roll_deg",
                        "gaze_zone",
                        "reprojection_error_px",
                        "quality",
                    ]
                )
            quality_label = (
                _quality_from_error(reproj_err) if reproj_err is not None else "n/a"
            )
            writer.writerow(
                [
                    datetime.now().isoformat(),
                    "image",
                    str(image_path),
                    method,
                    f"{result.pitch_deg:.4f}",
                    f"{result.yaw_deg:.4f}",
                    f"{result.roll_deg:.4f}",
                    zone,
                    f"{reproj_err:.4f}" if reproj_err is not None else "",
                    quality_label,
                ]
            )

    return result, image


def _ema(prev: np.ndarray | None, cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return cur
    return (1.0 - alpha) * prev + alpha * cur


def _run_webcam(
    method: str,
    camera_index: int,
    logger: logging.Logger,
    alpha: float,
    record_csv: str | None = None,
    attention_summary: bool = False,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Failed to open webcam index %d", camera_index)
        raise SystemExit(1)

    facemesh = None
    if method in ("mediapipe", "auto", "hybrid"):
        try:
            from mediapipe import solutions as mp_solutions
            FaceMesh = mp_solutions.face_mesh.FaceMesh
            facemesh = FaceMesh(
                static_image_mode=False,
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            if method == "mediapipe":
                logger.error("MediaPipe unavailable; cannot run mediapipe method")
                raise SystemExit(1)
            else:
                logger.warning("MediaPipe unavailable; falling back to heuristic method")
                method = "heuristic"

    prev_rvec: np.ndarray | None = None
    prev_tvec: np.ndarray | None = None
    prev_angles: np.ndarray | None = None

    offset_pitch = 0.0
    offset_yaw = 0.0
    offset_roll = 0.0

    csv_file = None
    csv_writer = None
    if record_csv is not None:
        csv_path = Path(record_csv)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "timestamp",
                "mode",
                "frame_index",
                "method",
                "pitch_deg",
                "yaw_deg",
                "roll_deg",
                "gaze_zone",
                "reprojection_error_px",
                "quality",
            ]
        )

    logger.info("Press 'q' to quit webcam mode")

    frame_idx = 0
    session_start = time.time()
    last_zone: str | None = None
    last_zone_ts = session_start
    zone_durations: dict[str, float] = {
        "CENTER": 0.0,
        "LEFT": 0.0,
        "RIGHT": 0.0,
        "UP": 0.0,
        "DOWN": 0.0,
        "UNKNOWN": 0.0,
    }
    zone_switches = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read from webcam")
                break

            try:
                if method == "heuristic":
                    result = estimate_head_pose_heuristic(frame)
                elif method == "mediapipe":
                    result = estimate_head_pose_mediapipe(frame, facemesh=facemesh)
                elif method == "hybrid":
                    result = estimate_head_pose_hybrid(frame, facemesh=facemesh)
                else:
                    if facemesh is not None:
                        result = estimate_head_pose_mediapipe(frame, facemesh=facemesh)
                    else:
                        result = estimate_head_pose_heuristic(frame)
            except HeadPoseError as exc:
                logger.warning("Pose estimation error: %s", exc)
                continue
            except Exception:
                logger.exception("Unexpected error during pose estimation")
                continue

            if alpha > 0.0:
                if result.rvec is not None and result.tvec is not None:
                    rvec = _ema(prev_rvec, result.rvec.astype(float), alpha)
                    tvec = _ema(prev_tvec, result.tvec.astype(float), alpha)
                    prev_rvec, prev_tvec = rvec, tvec
                    pitch_deg, yaw_deg, roll_deg = euler_from_rvec(rvec)
                    result = HeadPoseResult(
                        pitch_deg=pitch_deg,
                        yaw_deg=yaw_deg,
                        roll_deg=roll_deg,
                        nose_point=result.nose_point,
                        rvec=rvec,
                        tvec=tvec,
                        camera_matrix=result.camera_matrix,
                        dist_coeffs=result.dist_coeffs,
                    )
                else:
                    angles = np.array([result.pitch_deg, result.yaw_deg, result.roll_deg], dtype=float)
                    angles = _ema(prev_angles, angles, alpha)
                    prev_angles = angles
                    result = HeadPoseResult(
                        pitch_deg=float(angles[0]),
                        yaw_deg=float(angles[1]),
                        roll_deg=float(angles[2]),
                        nose_point=result.nose_point,
                    )

            cal_pitch = result.pitch_deg - offset_pitch
            cal_yaw = result.yaw_deg - offset_yaw
            cal_roll = result.roll_deg - offset_roll

            overlay = draw_pose_overlay(frame, result)
            zone = _classify_gaze(cal_pitch, cal_yaw)
            reproj_err = getattr(result, "reprojection_error", None)
            quality = _quality_from_error(reproj_err) if reproj_err is not None else "n/a"

            if attention_summary:
                now = time.time()
                if last_zone is None:
                    last_zone = zone
                    last_zone_ts = now
                else:
                    dt = now - last_zone_ts
                    zone_durations[last_zone] += max(dt, 0.0)
                    if zone != last_zone:
                        zone_switches += 1
                    last_zone = zone
                    last_zone_ts = now

            if csv_writer is not None:
                csv_writer.writerow(
                    [
                        datetime.now().isoformat(),
                        "webcam",
                        frame_idx,
                        method,
                        f"{cal_pitch:.4f}",
                        f"{cal_yaw:.4f}",
                        f"{cal_roll:.4f}",
                        zone,
                        f"{reproj_err:.4f}" if reproj_err is not None else "",
                        quality,
                    ]
                )

            cv2.putText(
                overlay,
                f"P {cal_pitch:.1f} | Y {cal_yaw:.1f} | R {cal_roll:.1f} | {zone}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            if reproj_err is not None:
                cv2.putText(
                    overlay,
                    f"Quality: {quality} ({reproj_err:.1f} px)",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Gaze Estimation", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                offset_pitch = result.pitch_deg
                offset_yaw = result.yaw_deg
                offset_roll = result.roll_deg
                logger.info(
                    "Neutral pose calibrated at P=%.2f, Y=%.2f, R=%.2f",
                    offset_pitch,
                    offset_yaw,
                    offset_roll,
                )

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if facemesh is not None:
            facemesh.close()
        if csv_file is not None:
            csv_file.close()
        if attention_summary:
            end_ts = time.time()
            if last_zone is not None:
                dt = end_ts - last_zone_ts
                zone_durations[last_zone] += max(dt, 0.0)
            total = sum(zone_durations.values())
            logger.info("Attention summary (%.1f s):", total)
            if total > 0:
                for z, t in zone_durations.items():
                    pct = (t / total) * 100.0
                    logger.info("  %s: %.2f s (%.1f%%)", z, t, pct)
            logger.info("Zone switches: %d", zone_switches)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("gaze_estimator.cli")

    if args.webcam:
        _run_webcam(
            args.method,
            args.camera,
            logger,
            args.stabilize,
            args.record_csv,
            args.attention_summary,
        )
        return

    if not args.image:
        logger.error("Please provide an image path, or use --webcam mode")
        raise SystemExit(1)

    image_path = Path(args.image)
    if not image_path.is_file():
        logger.error("Image file does not exist: %s", image_path)
        raise SystemExit(1)

    result, image = _estimate_single_image(image_path, args.method, logger, args.record_csv)

    if args.output is not None:
        overlay = draw_pose_overlay(image, result)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), overlay):
            logger.error("Failed to write overlay image to %s", output_path)
            raise SystemExit(1)
        logger.info("Overlay image saved to %s", output_path)
