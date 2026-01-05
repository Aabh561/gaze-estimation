# Gaze Estimation Project – AI Assistance Notes

## 1. Problem and requirements (in my own words)

The goal is to monitor human focus by checking if a person is looking in the right direction. Examples:
- A factory worker looking at the part they’re assembling.
- A truck driver looking at the road.

A simple way to measure this is to estimate **head orientation** in 3D:
- **Pitch**: nodding up and down.
- **Yaw**: turning left and right.
- **Roll**: tilting the head sideways (less important here).

The assignment:

- Build a **Python application** that:
  - takes a single image (720p+, PNG/JPEG, one face),
  - returns numerical estimates of pitch and yaw (and optionally roll).
- Target accuracy: roughly **±5°** (goal, not guaranteed in my current fallback).
- The solution must be **generic**:
  - no training per person,
  - no registering a specific face before use.
- Use:
  - **Python 3.12+**
  - **uv** for dependency management
  - **OpenCV** (MediaPipe is optional)
- Constraints:
  - Handle errors with logging and exceptions.
  - Log to **console + timestamped log file**.
  - Never fail silently.
  - Maintain:
    - `research.md` with resources I used.
    - `.ai/` folder with AI-related notes.
    - `README.md` with setup + explanation.
  - Make regular Git commits.
  - Try to finish by December 8.

## 2. High-level solution approach

I used AI mainly to help me understand the problem and design a plan. The core ideas:

1. **CLI app**:
   - A command `gaze-estimate` that:
     - takes an image path,
     - optionally an output path for an overlay,
     - runs head pose estimation,
     - prints pitch/yaw/roll,
     - saves a visual overlay.

2. **Head pose estimation (ideal design)**:
   - Use **MediaPipe FaceMesh** to get a dense set of 3D face landmarks.
   - Select a small subset of stable points:
     - eye corners, nose tip, mouth corners, chin.
   - Build **2D–3D correspondences**:
     - 2D: pixel coordinates in the image.
     - 3D: approximate head model coordinates.
   - Use **OpenCV `solvePnP`** to solve the Perspective-n-Point problem:
     - get a rotation vector (`rvec`) and translation vector (`tvec`).
   - Convert the rotation to **pitch, yaw, roll** using `cv2.RQDecomp3x3`.
   - Return the angles and optionally project a 3D axis onto the face for visualization.

3. **Logging + robustness**:
   - Central `setup_logging()`:
     - creates `logs/` and a timestamped file,
     - logs to both console and file with timestamps/levels.
   - Validate:
     - input image path exists,
     - image can be read,
     - face is detected.
   - Use a custom `HeadPoseError` for predictable failures.

4. **Evaluation**:
   - Use several images of myself at different poses.
   - Check qualitatively:
     - turning right increases yaw,
     - turning left decreases yaw,
     - moving head up/down changes pitch with the correct sign.
   - Later, I could compare approximate angles to known camera/head poses.

## 3. What actually happened with MediaPipe

I initially implemented the MediaPipe + `solvePnP` pipeline in `head_pose_estimator.py`, but on my Windows + Python 3.12.4 environment I hit this error at import time:

> `ImportError: DLL load failed while importing _framework_bindings: A dynamic link library (DLL) initialization routine failed.`

Key points:

- The error occurred **inside** the Mediapipe package when importing its compiled DLLs.
- Reinstalling Mediapipe in the virtualenv did not fix it.
- The crash happened **before** my own estimator code could run.
- Conclusion: this is a **native dependency / compatibility** issue with Mediapipe’s Windows wheel on my system, not a bug in my Python logic.

Because I needed a **working demo**, I decided to remove Mediapipe from this repo and design an **OpenCV-only fallback** that still follows the constraints (Python 3.12, OpenCV, no training).

## 4. Final implementation (OpenCV-only fallback)

### 4.1 CLI (`src/gaze_estimator/cli.py`)

Responsibilities:

- Parse command-line arguments:
  - required: `image` path,
  - optional: `--output/-o` for overlay image path,
  - optional: `--log-level`.
- Call `setup_logging(log_level)` so logs go to console and `logs/app_YYYYMMDD_HHMMSS.log`.
- Validate the image path and read it with `cv2.imread`.
- Call `estimate_head_pose(image)` from `head_pose_estimator.py`.
- Print angles:
  - `Pitch: X°`, `Yaw: Y°`, `Roll: Z°`.
- If `--output` is provided:
  - call `draw_pose_overlay(image, result)`,
  - save the overlay image,
  - log success/failure.

### 4.2 Logging (`src/gaze_estimator/logging_config.py`)

Responsibilities:

- Find the project root and create a `logs/` directory.
- Build a log filename with a timestamp.
- Configure a root logger:
  - console handler at the requested level (INFO, DEBUG, etc.),
  - file handler at DEBUG level (everything),
  - consistent format: timestamp, level, logger name, message.
- Remove old handlers to avoid duplicate logs.

### 4.3 Head pose estimator (`src/gaze_estimator/head_pose_estimator.py`)

#### a) `HeadPoseError` and `HeadPoseResult`

- `HeadPoseError`: custom exception for predictable failures (invalid image, no face, etc.).
- `HeadPoseResult`:
  - `pitch_deg`, `yaw_deg`, `roll_deg` – approximate angles.
  - `nose_point` – the face center where I draw the overlay.
  - Optional 3D fields (`rvec`, `tvec`, etc.) kept for API compatibility, but not used in the fallback.

#### b) `_detect_face_bbox(image)`

- Convert the image to grayscale.
- Use OpenCV’s built-in **Haar cascade**:
  - `haarcascade_frontalface_default.xml`.
- Detect faces with `detectMultiScale`.
- If no faces → raise `HeadPoseError("No face detected")`.
- If multiple faces → pick the largest bounding box (assume main subject).
- Returns `(x, y, w, h)` for the selected face.

#### c) `_approximate_pose_from_bbox(image, bbox)`

Idea: **use where the face appears in the frame to approximate angles**.

Steps:

1. Compute center of the face box:
   - `cx = x + w/2`, `cy = y + h/2`.
2. Normalize offsets relative to image center:
   - `nx` in [-1, 1] horizontally,
   - `ny` in [-1, 1] vertically.
3. Map these to angles:
   - `yaw_deg = nx * max_yaw`, with `max_yaw ≈ 30°`.
   - `pitch_deg = -ny * max_pitch`, with `max_pitch ≈ 20°`.
   - `roll_deg = 0` (not modeled in this version).
4. Set `nose_point = (cx, cy)` and return a `HeadPoseResult`.

This gives a smooth, deterministic mapping:
- Face more to the right → positive yaw.
- Face more to the left → negative yaw.
- Face higher in the frame → looking up (negative pitch).
- Face lower → looking down (positive pitch).

It is **approximate**, not as accurate as a full landmark + PnP approach, but it:
- requires only OpenCV,
- runs reliably in my environment,
- demonstrates the core idea of angle estimation.

#### d) `estimate_head_pose(image)`

- Validate the image is not `None` or empty.
- Call `_detect_face_bbox(image)` to get the face bounding box.
- Call `_approximate_pose_from_bbox(image, bbox)` to get angles.
- Return `HeadPoseResult`.

#### e) `draw_pose_overlay(image, result)`

- Copy the image.
- Draw a small yellow dot at `nose_point`.
- Compute arrow endpoint based on `result.yaw_deg` and `result.pitch_deg`:
  - normalize angles to [-1, 1],
  - scale to a fixed length.
- Draw a red arrow from the nose point to the endpoint.
- Return the overlay image.

## 5. How I used AI during this project

I used AI for:

- **Understanding the domain**:
  - Meaning of pitch, yaw, roll.
  - Typical pipelines for head pose estimation (landmarks + PnP).
- **Design help**:
  - Structuring the project into modules (`cli`, `head_pose_estimator`, `logging_config`).
  - Designing the initial MediaPipe + `solvePnP` solution.
- **Debugging help**:
  - Interpreting the Mediapipe `_framework_bindings` DLL error.
  - Deciding on an OpenCV-only fallback that still fits the assignment.
- **Code explanation**:
  - Having the assistant explain each part of the code in simple terms so I could restate it in my own words.

I did **not**:
- blindly paste large chunks of code I didn’t understand,
- leave comments or docs that don’t match the code.

I **can** explain:
- what each file does,
- how the approximate angles are computed from the face position,
- why the current version doesn’t use Mediapipe,
- how I’d improve accuracy later with landmarks + `solvePnP` in a better environment.

## 6. Future improvements

If I had more time or a different environment (e.g. Linux container):

1. Re-enable the full landmark-based pipeline:
   - MediaPipe FaceMesh → 468 landmarks.
   - Select stable keypoints.
   - SolvePnP → precise 3D rotation.
   - Convert to pitch, yaw, roll.

2. Add a simple evaluation:
   - Capture images with roughly known head poses (e.g. ±15°, ±30°).
   - Compute average error compared to rough ground truth.

3. Make the approximate OpenCV-only method a configurable fallback:
   - `--mode mediapipe` vs `--mode opencv_fallback`.
