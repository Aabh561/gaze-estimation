from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp  # type: ignore
    _MP_AVAILABLE = True
except Exception:
    mp = None  # type: ignore
    _MP_AVAILABLE = False


class HeadPoseError(Exception):
    """Head pose estimation error."""


@dataclass
class HeadPoseResult:
    """Head pose result container."""

    pitch_deg: float
    yaw_deg: float
    roll_deg: float
    nose_point: Tuple[int, int]
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None
    reprojection_error: Optional[float] = None


def _detect_face_bbox(image: np.ndarray) -> Tuple[int, int, int, int]:
    if image is None or image.size == 0:
        raise HeadPoseError("Invalid image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise HeadPoseError("Failed to load Haar cascade")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        raise HeadPoseError("No face detected")
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def _approximate_pose_from_bbox(
    image: np.ndarray, bbox: Tuple[int, int, int, int]
) -> HeadPoseResult:
    h, w, _ = image.shape
    x, y, bw, bh = bbox
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    nx = (cx - w / 2.0) / (w / 2.0)
    ny = (cy - h / 2.0) / (h / 2.0)
    max_yaw = 30.0
    max_pitch = 20.0
    yaw_deg = float(nx * max_yaw)
    pitch_deg = float(-ny * max_pitch)
    roll_deg = 0.0
    nose_point = (int(cx), int(cy))
    return HeadPoseResult(pitch_deg=pitch_deg, yaw_deg=yaw_deg, roll_deg=roll_deg, nose_point=nose_point)


def estimate_head_pose_heuristic(image: np.ndarray) -> HeadPoseResult:
    bbox = _detect_face_bbox(image)
    return _approximate_pose_from_bbox(image, bbox)


_MODEL_POINTS_3D = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1],
    ],
    dtype=np.float64,
)

_LANDMARK_IDS = {
    "nose": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}


def _compute_camera_matrix(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    focal_length = w
    center = (w / 2.0, h / 2.0)
    return np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )


def _landmarks_to_image_points(landmarks, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = image.shape[:2]
    pts = []
    for key in ("nose", "chin", "left_eye", "right_eye", "mouth_left", "mouth_right"):
        idx = _LANDMARK_IDS[key]
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append([x, y])
    nose_pt = (pts[0][0], pts[0][1])
    return np.array(pts, dtype=np.float64), nose_pt


def _rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        roll = 0.0
    return float(pitch), float(yaw), float(roll)


def euler_from_rvec(rvec: np.ndarray) -> Tuple[float, float, float]:
    R, _ = cv2.Rodrigues(rvec)
    return _rotation_matrix_to_euler_angles(R)


def estimate_head_pose_mediapipe(image: np.ndarray, facemesh=None) -> HeadPoseResult:
    if not _MP_AVAILABLE:
        raise HeadPoseError("MediaPipe unavailable")
    if image is None or image.size == 0:
        raise HeadPoseError("Invalid image")
    FaceMesh = mp.solutions.face_mesh.FaceMesh  # type: ignore
    owns_facemesh = False
    if facemesh is None:
        facemesh = FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        owns_facemesh = True
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = facemesh.process(rgb)
        if not results.multi_face_landmarks:
            raise HeadPoseError("No face landmarks detected")
        landmarks = results.multi_face_landmarks[0].landmark
        image_points, nose_pt = _landmarks_to_image_points(landmarks, image)
        camera_matrix = _compute_camera_matrix(image)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            raise HeadPoseError("solvePnP failed")
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                _MODEL_POINTS_3D,
                image_points,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
            )
        except Exception:
            pass

        try:
            projected_points, _ = cv2.projectPoints(
                _MODEL_POINTS_3D,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            )
            projected_points_2d = projected_points.reshape(-1, 2)
            reprojection_error = float(
                np.mean(
                    np.linalg.norm(projected_points_2d - image_points, axis=1)
                )
            )
        except Exception:
            reprojection_error = None

        pitch_deg, yaw_deg, roll_deg = euler_from_rvec(rvec)
        return HeadPoseResult(
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            roll_deg=roll_deg,
            nose_point=nose_pt,
            rvec=rvec,
            tvec=tvec,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            reprojection_error=reprojection_error,
        )
    finally:
        if owns_facemesh:
            facemesh.close()


def estimate_head_pose_hybrid(
    image: np.ndarray,
    facemesh=None,
    reproj_threshold: float = 7.0,
) -> HeadPoseResult:
    """Hybrid estimator: use MediaPipe when quality is good, else heuristic.

    - If MediaPipe is available and succeeds, and the reprojection error is
      below ``reproj_threshold``, return the MediaPipe-based result.
    - Otherwise, fall back to the heuristic estimator.
    """
    if image is None or image.size == 0:
        raise HeadPoseError("Invalid image")

    if _MP_AVAILABLE:
        try:
            mp_result = estimate_head_pose_mediapipe(image, facemesh=facemesh)
            err = getattr(mp_result, "reprojection_error", None)
            if err is not None and err <= reproj_threshold:
                return mp_result
        except Exception:
            pass

    return estimate_head_pose_heuristic(image)


def estimate_head_pose(image: np.ndarray) -> HeadPoseResult:
    if image is None or image.size == 0:
        raise HeadPoseError("Invalid image")
    if _MP_AVAILABLE:
        try:
            return estimate_head_pose_mediapipe(image)
        except Exception:
            pass
    return estimate_head_pose_heuristic(image)


def draw_pose_overlay(image: np.ndarray, result: HeadPoseResult) -> np.ndarray:
    overlay = image.copy()
    nose_point = tuple(result.nose_point)

    color = (0, 255, 255)
    reproj_err = getattr(result, "reprojection_error", None)
    if reproj_err is not None:
        if reproj_err < 3.0:
            color = (0, 255, 0)
        elif reproj_err > 7.0:
            color = (0, 0, 255)
    cv2.circle(overlay, nose_point, 4, color, -1)

    if result.rvec is not None and result.camera_matrix is not None:
        axis_len = 60.0
        axis_3d = np.array(
            [[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]],
            dtype=np.float64,
        )
        dist = result.dist_coeffs if result.dist_coeffs is not None else np.zeros((4, 1))
        img_pts, _ = cv2.projectPoints(axis_3d, result.rvec, result.tvec, result.camera_matrix, dist)
        p0 = tuple(img_pts[0].ravel().astype(int))
        pX = tuple(img_pts[1].ravel().astype(int))
        pY = tuple(img_pts[2].ravel().astype(int))
        pZ = tuple(img_pts[3].ravel().astype(int))
        cv2.line(overlay, p0, pX, (0, 0, 255), 2)
        cv2.line(overlay, p0, pY, (0, 255, 0), 2)
        cv2.line(overlay, p0, pZ, (255, 0, 0), 2)
        return overlay

    length = 60.0
    nx = max(-1.0, min(1.0, result.yaw_deg / 30.0))
    ny = max(-1.0, min(1.0, result.pitch_deg / 20.0))
    end_x = int(nose_point[0] + -nx * length)
    end_y = int(nose_point[1] + ny * length)
    cv2.arrowedLine(overlay, nose_point, (end_x, end_y), color, 2, tipLength=0.25)
    return overlay
