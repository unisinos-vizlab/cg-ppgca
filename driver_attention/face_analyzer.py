"""
Face detection and head pose estimation via MediaPipe FaceMesh.

Maps webcam frames to openpilot-compatible driver state data using
solvePnP for head pose and Eye Aspect Ratio (EAR) for blink detection.
"""
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FaceData:
    """Analyzed driver face state for a single frame."""
    detected: bool = False
    face_prob: float = 0.0
    # Head pose angles in radians (openpilot convention)
    pitch: float = 0.0          # positive = tilt forward/down
    yaw: float = 0.0            # positive = turn right
    orientation_std: float = 1.0
    # Blink: 0.0 = open, 1.0 = closed
    blink_left: float = 0.0
    blink_right: float = 0.0
    eye_left_prob: float = 0.0
    eye_right_prob: float = 0.0
    sunglasses_prob: float = 0.0
    phone_prob: float = 0.0
    # Eye Aspect Ratios for display
    ear_left: float = 0.3
    ear_right: float = 0.3
    # Visualization data
    landmarks: object = None
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    frame_h: int = 480
    frame_w: int = 640


class FaceAnalyzer:
    """
    Driver face analyzer using MediaPipe FaceMesh + OpenCV solvePnP.

    Produces head pose (pitch/yaw) and blink probabilities that match the
    scale expected by DriverAttentionMonitor (ported from openpilot).
    """

    # Generic 6-point 3D face model (mm), same landmark mapping as classic
    # head pose papers (Kazemi 2014, Guo 2020).
    FACE_3D = np.array([
        (  0.0,    0.0,    0.0),   # nose tip         → lm[4]
        (  0.0,  -63.6,  -12.5),  # chin             → lm[152]
        (-43.3,   32.7,  -26.0),  # left eye outer   → lm[263]
        ( 43.3,   32.7,  -26.0),  # right eye outer  → lm[33]
        (-28.9,  -28.9,  -24.1),  # left mouth       → lm[287]
        ( 28.9,  -28.9,  -24.1),  # right mouth      → lm[57]
    ], dtype=np.float64)
    FACE_LM_IDS = [4, 152, 263, 33, 287, 57]

    # 6-point eye rings for EAR (top-left, top-center, top-right,
    # bottom-right, bottom-center, bottom-left — going clockwise/CCW)
    LEFT_EYE  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33,  160, 158, 133, 153, 144]

    def __init__(self) -> None:
        mp_fm = mp.solutions.face_mesh
        self._mesh = mp_fm.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._prev_rvec: Optional[np.ndarray] = None
        self._prev_tvec: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ear(lm, ids: list, w: int, h: int) -> float:
        """Eye Aspect Ratio from 6 evenly-spaced lid landmarks."""
        p = np.array([[lm[i].x * w, lm[i].y * h] for i in ids])
        A = np.linalg.norm(p[1] - p[5])
        B = np.linalg.norm(p[2] - p[4])
        C = np.linalg.norm(p[0] - p[3])
        return float((A + B) / (2.0 * C + 1e-9))

    @staticmethod
    def _blink_prob(ear: float) -> float:
        """
        Convert EAR to blink probability via sigmoid.
        EAR ~0.25 → open (prob≈0.12), EAR ~0.15 → closed (prob≈0.88).
        Threshold of 0.865 from openpilot BLINK_THRESHOLD triggers at EAR≈0.155.
        """
        return float(1.0 / (1.0 + np.exp(40.0 * (ear - 0.20))))

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray):
        """ZYX Euler decomposition: returns (pitch, yaw, roll) in radians."""
        sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
        if sy > 1e-6:
            pitch = float(np.arctan2(R[2, 1], R[2, 2]))
            yaw   = float(np.arctan2(-R[2, 0], sy))
        else:
            pitch = float(np.arctan2(-R[1, 2], R[1, 1]))
            yaw   = float(np.arctan2(-R[2, 0], sy))
        return pitch, yaw

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, frame: np.ndarray) -> FaceData:
        """Return FaceData for one BGR camera frame."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._prev_rvec = None
            self._prev_tvec = None
            return FaceData(frame_h=h, frame_w=w)

        lm = results.multi_face_landmarks[0].landmark

        img_pts = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in self.FACE_LM_IDS],
            dtype=np.float64,
        )

        # Approximate pinhole camera (no distortion)
        fl = float(w)
        cam_mat = np.array(
            [[fl, 0, w / 2.0], [0, fl, h / 2.0], [0, 0, 1]],
            dtype=np.float64,
        )
        dist = np.zeros((4, 1))

        ok, rvec, tvec = cv2.solvePnP(
            self.FACE_3D, img_pts, cam_mat, dist,
            rvec=self._prev_rvec,
            tvec=self._prev_tvec,
            useExtrinsicGuess=self._prev_rvec is not None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return FaceData(
                detected=True, face_prob=0.90,
                orientation_std=0.8, landmarks=lm,
                frame_h=h, frame_w=w,
            )

        self._prev_rvec = rvec.copy()
        self._prev_tvec = tvec.copy()

        R, _ = cv2.Rodrigues(rvec)
        pitch, yaw = self._rotation_matrix_to_euler(R)

        ear_l = self._ear(lm, self.LEFT_EYE,  w, h)
        ear_r = self._ear(lm, self.RIGHT_EYE, w, h)

        return FaceData(
            detected=True,
            face_prob=0.95,
            pitch=pitch,
            yaw=yaw,
            orientation_std=0.05,
            blink_left=self._blink_prob(ear_l),
            blink_right=self._blink_prob(ear_r),
            eye_left_prob=0.95,
            eye_right_prob=0.95,
            sunglasses_prob=0.0,
            phone_prob=0.0,
            ear_left=ear_l,
            ear_right=ear_r,
            landmarks=lm,
            rvec=rvec,
            tvec=tvec,
            frame_h=h,
            frame_w=w,
        )
