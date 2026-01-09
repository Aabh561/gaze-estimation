# Gaze Estimation Toy Project

This project estimates head orientation (pitch, yaw, roll) from an image or webcam. It now prefers a landmark-based pipeline (MediaPipe FaceMesh + OpenCV solvePnP) when available, with a robust OpenCV-only fallback.

---

## What’s New

- Landmark-based 3D pose via **MediaPipe FaceMesh + solvePnP**
- Quality-aware overlay: 3D axes when full pose is available; arrow otherwise, color-coded by pose quality,
- CLI upgrades: single image or **webcam mode**, method selection (`auto|mediapipe|heuristic|hybrid`)
- Gaze zone classification (`CENTER/LEFT/RIGHT/UP/DOWN`) from pitch/yaw
- Hybrid estimator that chooses between MediaPipe and heuristic based on reprojection error
- Interactive neutral-pose calibration in webcam mode (`c` key)
- Optional CSV logging of pose, gaze zone, and quality over time
- Webcam attention summary: dwell time per gaze zone and number of gaze switches (`--attention-summary`)
- Evaluation helper exposed as `gaze-eval` with sample `eval_cases.json` and method comparison (`--compare-methods`)
- `test_data/` scaffold and guidance to capture labeled angles

---

## Installation

Use **uv** or pip:

```bash
# with uv (recommended)
uv pip install -e .

# or with pip
python -m pip install -e .
```

Dependencies include `opencv-python`, `numpy`, and `mediapipe`.

### Windows prerequisite (fix Mediapipe DLL error)
If you previously saw `ImportError: DLL load failed while importing _framework_bindings`, install the **Microsoft Visual C++ Redistributable** (x64) for your system:
- https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

After installing, re-run installation. MediaPipe should import cleanly on Python 3.12.

If issues persist, alternatives:
- Use Python 3.10/3.11 or run in **WSL2/Docker** where prebuilt wheels are stable.
- Fall back to the OpenCV heuristic by running with `--method heuristic`.

---

## Usage

### Single image
```bash
# auto: prefer mediapipe, fallback to heuristic
uv run gaze-estimate path/to/image.jpg -o out/overlay.jpg --method auto

# force mediapipe (will error if unavailable)
uv run gaze-estimate path/to/image.jpg --method mediapipe

# force heuristic
uv run gaze-estimate path/to/image.jpg --method heuristic

# hybrid: use mediapipe when quality (reprojection error) is good, else heuristic
uv run gaze-estimate path/to/image.jpg --method hybrid
```

Outputs printed angles and optionally saves an overlay image.

### Webcam mode
```bash
uv run gaze-estimate --webcam --method auto --stabilize 0.4
```
- Press `q` to quit.
- Press `c` to capture your current pose as neutral; subsequent angles are relative to this.
- If MediaPipe is available, it uses landmarks; otherwise it falls back to heuristic.

#### Optional CSV logging
```bash
uv run gaze-estimate --webcam --method hybrid --stabilize 0.4 --record-csv logs/webcam_session.csv --attention-summary
```
Each frame logs timestamp, method, calibrated pitch/yaw/roll, gaze zone, and pose quality. With `--attention-summary`, the app also prints total dwell time and percentage spent in each gaze zone at the end of the session.

---

## Evaluation (MAE)

Add 3+ labeled photos to `test_data/` and update `eval_cases.json` accordingly, then run:
```bash
uv run gaze-eval eval_cases.json --method hybrid --compare-methods
```
This reports mean absolute error for pitch and yaw for the chosen primary method, and (with `--compare-methods`) also reports separate MAE for the pure MediaPipe and pure heuristic pipelines.

`eval_cases.json` example (provided at repo root):
```json
[
  { "image": "test_data/neutral_0deg.jpg",     "pitch_deg": 0.0,  "yaw_deg": 0.0 },
  { "image": "test_data/yaw_left_15deg.jpg",   "pitch_deg": 0.0,  "yaw_deg": -15.0 },
  { "image": "test_data/yaw_right_40deg.jpg",  "pitch_deg": 0.0,  "yaw_deg": 40.0 },
  { "image": "test_data/pitch_up_10deg.jpg",   "pitch_deg": -10.0, "yaw_deg": 0.0 },
  { "image": "test_data/pitch_down_10deg.jpg", "pitch_deg": 10.0,  "yaw_deg": 0.0 }
]
```

---

## How It Works

- **Landmarks + solvePnP** (preferred):
  - Detect 2D landmarks (nose tip, chin, eye corners, mouth corners) via MediaPipe FaceMesh.
  - Map to canonical 3D model points and solve for pose using `cv2.solvePnP`.
  - Convert rotation to Euler angles and draw projected axes.
  - Compute a reprojection error (in pixels) as a pose-quality score.

- **Heuristic fallback**:
  - Detect face via Haar cascade; map face center position to pitch/yaw.
  - Fast and dependency-light, but less accurate.

- **Hybrid estimator**:
  - Run the MediaPipe pipeline when available.
  - If reprojection error is below a threshold, use that result.
  - Otherwise, fall back to the heuristic estimate.

- **Gaze zone classification**:
  - Threshold calibrated pitch/yaw into coarse zones: `CENTER`, `LEFT`, `RIGHT`, `UP`, `DOWN`.

- **Interactive calibration (webcam)**:
  - Press `c` once while looking straight ahead; future angles are reported relative to this neutral pose.

- **Attention summary (webcam)**:
  - With `--attention-summary`, accumulate dwell time per gaze zone and report seconds and percentages for each zone, plus the number of zone switches.

---

## Limitations & Notes

- Accuracy depends on lighting, face visibility, and camera intrinsics; the default intrinsics are approximated from image size.
- Landmark indices are standard for FaceMesh but can vary slightly; the current set works well for typical cameras.
- For strict accuracy targets (±5°), calibrating camera intrinsics and using more precise 3D model points improves results.

---

## Development

- Logs go to `logs/`.
- CLI entry points:
  - `gaze-estimate` for inference (image/webcam)
  - `gaze-eval` for evaluation
- Code lives under `src/gaze_estimator/`.

Contributions and improvements are welcome.
