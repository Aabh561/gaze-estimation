import argparse
import json
import logging
from pathlib import Path

import cv2

from .logging_config import setup_logging
from .head_pose_estimator import (
    estimate_head_pose,
    estimate_head_pose_heuristic,
    estimate_head_pose_mediapipe,
    estimate_head_pose_hybrid,
    HeadPoseError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gaze-eval",
        description=(
            "Evaluate the head pose estimator on a small set of images with "
            "approximate ground-truth pitch/yaw angles."
        ),
    )
    parser.add_argument(
        "config",
        type=str,
        help=(
            "Path to a JSON file containing a list of test cases. "
            "Each entry should be an object with fields: "
            "'image' (path), 'pitch_deg' (float), 'yaw_deg' (float)."
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["auto", "mediapipe", "heuristic", "hybrid"],
        default="auto",
        help=(
            "Pose estimation method for primary evaluation: "
            "'auto' uses MediaPipe when available, otherwise heuristic."
        ),
    )
    parser.add_argument(
        "--compare-methods",
        action="store_true",
        help=(
            "Also compute separate MAE for mediapipe and heuristic methods "
            "(when available) in addition to the primary method."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def load_cases(config_path: Path) -> list[dict]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Config JSON must be a list of test case objects")
    cases: list[dict] = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {idx} is not an object")
        for key in ("image", "pitch_deg", "yaw_deg"):
            if key not in entry:
                raise ValueError(f"Entry {idx} missing required field '{key}'")
        cases.append(entry)
    return cases


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("gaze_estimator.evaluate")

    config_path = Path(args.config)
    try:
        cases = load_cases(config_path)
    except Exception:
        logger.exception("Failed to load evaluation config from %s", config_path)
        raise SystemExit(1)

    if not cases:
        logger.error("No test cases found in config %s", config_path)
        raise SystemExit(1)

    logger.info("Loaded %d test cases from %s", len(cases), config_path)

    abs_pitch_errors: list[float] = []
    abs_yaw_errors: list[float] = []

    mp_pitch_errors: list[float] = []
    mp_yaw_errors: list[float] = []
    h_pitch_errors: list[float] = []
    h_yaw_errors: list[float] = []

    for idx, case in enumerate(cases, start=1):
        image_path = Path(case["image"])
        gt_pitch = float(case["pitch_deg"])
        gt_yaw = float(case["yaw_deg"])

        if not image_path.is_file():
            logger.error("[%d] Image file does not exist: %s", idx, image_path)
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            logger.error("[%d] Failed to read image: %s", idx, image_path)
            continue

        try:
            if args.method == "heuristic":
                result = estimate_head_pose_heuristic(image)
            elif args.method == "mediapipe":
                result = estimate_head_pose_mediapipe(image)
            elif args.method == "hybrid":
                result = estimate_head_pose_hybrid(image)
            else:
                result = estimate_head_pose(image)
        except HeadPoseError as exc:
            logger.error(
                "[%d] Head pose estimation failed for %s: %s", idx, image_path, exc
            )
            continue
        except Exception:
            logger.exception(
                "[%d] Unexpected error during head pose estimation for %s",
                idx,
                image_path,
            )
            continue

        pred_pitch = float(result.pitch_deg)
        pred_yaw = float(result.yaw_deg)

        err_pitch = abs(pred_pitch - gt_pitch)
        err_yaw = abs(pred_yaw - gt_yaw)

        abs_pitch_errors.append(err_pitch)
        abs_yaw_errors.append(err_yaw)

        if args.compare_methods:
            try:
                mp_res = estimate_head_pose_mediapipe(image)
                mp_pitch = float(mp_res.pitch_deg)
                mp_yaw = float(mp_res.yaw_deg)
                mp_pitch_errors.append(abs(mp_pitch - gt_pitch))
                mp_yaw_errors.append(abs(mp_yaw - gt_yaw))
            except Exception:
                logger.warning(
                    "[%d] Mediapipe comparison failed for %s", idx, image_path
                )

            # Heuristic branch.
            try:
                h_res = estimate_head_pose_heuristic(image)
                h_pitch = float(h_res.pitch_deg)
                h_yaw = float(h_res.yaw_deg)
                h_pitch_errors.append(abs(h_pitch - gt_pitch))
                h_yaw_errors.append(abs(h_yaw - gt_yaw))
            except Exception:
                logger.warning(
                    "[%d] Heuristic comparison failed for %s", idx, image_path
                )

        print(
            f"[{idx}] {image_path} | "
            f"gt_pitch={gt_pitch:.2f}, pred_pitch={pred_pitch:.2f}, err={err_pitch:.2f} deg | "
            f"gt_yaw={gt_yaw:.2f}, pred_yaw={pred_yaw:.2f}, err={err_yaw:.2f} deg"
        )

    if not abs_pitch_errors or not abs_yaw_errors:
        logger.error("No successful evaluations; cannot compute summary error")
        raise SystemExit(1)

    mean_pitch_err = sum(abs_pitch_errors) / len(abs_pitch_errors)
    mean_yaw_err = sum(abs_yaw_errors) / len(abs_yaw_errors)

    print()
    print("Summary (mean absolute error) - primary method:")
    print(
        f"Pitch MAE: {mean_pitch_err:.2f} degrees over {len(abs_pitch_errors)} images"
    )
    print(f"Yaw   MAE: {mean_yaw_err:.2f} degrees over {len(abs_yaw_errors)} images")

    if args.compare_methods:
        print()
        print("Method comparison (mediapipe vs heuristic):")
        if mp_pitch_errors and mp_yaw_errors:
            mp_pitch_mae = sum(mp_pitch_errors) / len(mp_pitch_errors)
            mp_yaw_mae = sum(mp_yaw_errors) / len(mp_yaw_errors)
            print(
                f"Mediapipe  - Pitch MAE: {mp_pitch_mae:.2f} deg over {len(mp_pitch_errors)} images"
            )
            print(
                f"             Yaw   MAE: {mp_yaw_mae:.2f} deg over {len(mp_yaw_errors)} images"
            )
        else:
            print("Mediapipe  - no successful evaluations for comparison")

        if h_pitch_errors and h_yaw_errors:
            h_pitch_mae = sum(h_pitch_errors) / len(h_pitch_errors)
            h_yaw_mae = sum(h_yaw_errors) / len(h_yaw_errors)
            print(
                f"Heuristic  - Pitch MAE: {h_pitch_mae:.2f} deg over {len(h_pitch_errors)} images"
            )
            print(
                f"             Yaw   MAE: {h_yaw_mae:.2f} deg over {len(h_yaw_errors)} images"
            )
        else:
            print("Heuristic  - no successful evaluations for comparison")


if __name__ == "__main__":  # pragma: no cover
    main()
