"""
Driver Attention Monitor — ported from openpilot selfdrive/monitoring/policy.py.

Ref: https://github.com/commaai/openpilot/blob/master/selfdrive/monitoring/policy.py

Key differences from openpilot:
- No vehicle data (speed, gear, steering).  The monitor runs in "always-on"
  standalone mode so it never suspends awareness counting.
- Calibration window shortened to CALIB_FRAMES (≈5 s at 30 fps) vs openpilot's
  1-minute driving window.
- No wheeltouch fallback policy; vision-only.
- After a terminal alert the awareness resets after TERMINAL_HOLD_S seconds so
  the monitor keeps running.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .face_analyzer import FaceData


# ---------------------------------------------------------------------------
# Settings — values mirror openpilot defaults unless noted
# ---------------------------------------------------------------------------

class _Settings:
    # Alert timeouts (seconds) — Euro NCAP / openpilot standard
    ALERT_1_S: float = 3.0    # first warning (green → yellow)
    ALERT_2_S: float = 5.0    # second warning (yellow → orange)
    ALERT_3_S: float = 11.0   # terminal (orange → red)

    RECOVERY_FACTOR_MAX: float = 5.0
    RECOVERY_FACTOR_MIN: float = 1.25

    # Detection thresholds (from openpilot DRIVER_MONITOR_SETTINGS)
    FACE_THRESH: float = 0.70
    EYE_THRESH:  float = 0.65
    SG_THRESH:   float = 0.90    # sunglasses
    BLINK_THRESH: float = 0.865

    PHONE_THRESH: float = 0.50

    # Pose thresholds (radians)
    PITCH_THRESH: float = 0.3133   # ~18°
    YAW_THRESH:   float = 0.4020   # ~23°
    # Pre-calibration fallback thresholds (more lenient)
    PITCH_NATURAL_THRESH: float = 0.449
    # Natural offsets before per-session calibration
    PITCH_NATURAL_OFFSET: float = 0.011
    YAW_NATURAL_OFFSET:   float = 0.075

    # Model uncertainty
    HI_STD_THRESH: float = 0.30

    # Distraction signal filter time constant (s) — matches openpilot
    DISTRACTED_FILTER_TC: float = 0.25

    # Per-session pose calibration
    CALIB_FRAMES: int = 150     # ~5 s at 30 fps (openpilot uses 1200 @ 20 Hz)
    CALIB_MAX_FRAMES: int = 900  # stop deweighting after 30 s

    # Terminal alert display before reset
    TERMINAL_HOLD_S: float = 3.0


S = _Settings()


# ---------------------------------------------------------------------------
# State snapshot returned each frame
# ---------------------------------------------------------------------------

@dataclass
class AttentionState:
    awareness: float            # 0.0–1.0  (below 0 clamped to 0 for alerts)
    alert_level: int            # 0=none 1=yellow 2=orange 3=red
    face_detected: bool
    is_distracted: bool
    distracted_types: Dict[str, bool]
    calibrated: bool
    calib_percent: int          # 0–100
    pitch: float
    yaw: float
    blink_left: float
    blink_right: float
    terminal_count: int         # cumulative terminal alerts
    distraction_filter: float   # smoothed distraction signal


# ---------------------------------------------------------------------------
# Main monitor class
# ---------------------------------------------------------------------------

class DriverAttentionMonitor:
    """
    Tracks driver attention level from FaceData frames.

    Usage::

        monitor = DriverAttentionMonitor()
        while capturing:
            face = analyzer.analyze(frame)
            state = monitor.update(face, dt)
    """

    def __init__(self) -> None:
        self._awareness: float = 1.0
        self._distraction_filter: float = 0.0  # low-pass smoothed distraction
        self._face_detected: bool = False
        self._driver_distracted: bool = False
        self._distracted_types: Dict[str, bool] = {}

        # Head pose
        self._pitch: float = 0.0
        self._yaw: float = 0.0
        self._low_std: bool = True
        self._blink_left: float = 0.0
        self._blink_right: float = 0.0
        self._phone_prob: float = 0.0

        # Per-session calibration
        self._pitch_offset: float = S.PITCH_NATURAL_OFFSET
        self._yaw_offset: float = S.YAW_NATURAL_OFFSET
        self._calib_samples: int = 0

        # Alert bookkeeping
        self._terminal_count: int = 0
        self._terminal_hold_elapsed: float = 0.0
        self._in_terminal: bool = False

        # Derived thresholds
        self._thresh1 = 1.0 - S.ALERT_1_S / S.ALERT_3_S   # ≈ 0.727
        self._thresh2 = 1.0 - S.ALERT_2_S / S.ALERT_3_S   # ≈ 0.545

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset awareness and calibration to initial state."""
        self.__init__()

    # ------------------------------------------------------------------
    # Internal helpers — logic ported from openpilot policy.py
    # ------------------------------------------------------------------

    def _update_face_state(self, face: FaceData, dt: float) -> None:
        """Ingest FaceData and calibrate pose offsets."""
        self._face_detected = face.face_prob > S.FACE_THRESH and face.detected
        self._pitch = face.pitch
        self._yaw = face.yaw
        self._low_std = face.orientation_std < S.HI_STD_THRESH

        # Build blink values: zero out if eye not detected or sunglasses
        eye_l = face.eye_left_prob > S.EYE_THRESH and face.sunglasses_prob < S.SG_THRESH
        eye_r = face.eye_right_prob > S.EYE_THRESH and face.sunglasses_prob < S.SG_THRESH
        self._blink_left  = face.blink_left  * float(eye_l)
        self._blink_right = face.blink_right * float(eye_r)
        self._phone_prob  = face.phone_prob

        # Incremental mean for pose calibration
        if self._face_detected and self._low_std and not self._driver_distracted:
            n = min(self._calib_samples + 1, S.CALIB_MAX_FRAMES)
            alpha = 1.0 / n
            self._pitch_offset = (1.0 - alpha) * self._pitch_offset + alpha * face.pitch
            self._yaw_offset   = (1.0 - alpha) * self._yaw_offset   + alpha * face.yaw
            self._calib_samples = min(self._calib_samples + 1, S.CALIB_MAX_FRAMES)

    def _get_distracted_types(self) -> None:
        """
        Determine which distraction types are active.
        Directly mirrors openpilot's _get_distracted_types().
        """
        calibrated = self._calib_samples >= S.CALIB_FRAMES

        if calibrated:
            pitch_err = self._pitch - self._pitch_offset
            yaw_err   = self._yaw   - self._yaw_offset
            pitch_thr = S.PITCH_THRESH
        else:
            pitch_err = self._pitch - S.PITCH_NATURAL_OFFSET
            yaw_err   = self._yaw   - S.YAW_NATURAL_OFFSET
            pitch_thr = S.PITCH_NATURAL_THRESH

        # No positive pitch limit — only looking "up" beyond natural triggers
        pitch_err = 0.0 if pitch_err > 0 else abs(pitch_err)
        yaw_err   = abs(yaw_err)

        self._distracted_types = {
            'pose':  (pitch_err > pitch_thr) or (yaw_err > S.YAW_THRESH),
            'eye':   (self._blink_left + self._blink_right) * 0.5 > S.BLINK_THRESH,
            'phone': self._phone_prob > S.PHONE_THRESH,
        }
        self._driver_distracted = (
            any(self._distracted_types.values())
            and self._face_detected
            and self._low_std
        )

    def _update_awareness(self, dt: float) -> None:
        """
        Decrement or restore awareness based on filtered distraction signal.
        Mirrors openpilot's _update_events() without vehicle-specific branches.
        """
        # First-order low-pass filter on distraction signal (openpilot param: 0.25 s)
        alpha = dt / (dt + S.DISTRACTED_FILTER_TC)
        self._distraction_filter += alpha * (float(self._driver_distracted) - self._distraction_filter)

        step = dt / S.ALERT_3_S

        certainly_distracted = (
            self._distraction_filter > 0.63
            and self._driver_distracted
            and self._face_detected
        )
        maybe_distracted = not self._face_detected

        if certainly_distracted or maybe_distracted:
            self._awareness = max(-0.1, self._awareness - step)
        elif (
            self._distraction_filter < 0.37
            and self._face_detected
            and self._low_std
        ):
            # Awareness recovers faster when nearly depleted (openpilot formula)
            recovery = (
                (S.RECOVERY_FACTOR_MAX - S.RECOVERY_FACTOR_MIN) * (1.0 - self._awareness)
                + S.RECOVERY_FACTOR_MIN
            ) * step
            self._awareness = min(1.0, self._awareness + recovery)

    def _handle_terminal(self, dt: float) -> None:
        """Hold terminal alert for TERMINAL_HOLD_S then auto-reset."""
        if self._awareness <= 0.0:
            if not self._in_terminal:
                self._terminal_count += 1
                self._in_terminal = True
            self._terminal_hold_elapsed += dt
            if self._terminal_hold_elapsed >= S.TERMINAL_HOLD_S:
                # Reset for next round
                self._awareness = 1.0
                self._distraction_filter = 0.0
                self._in_terminal = False
                self._terminal_hold_elapsed = 0.0
        else:
            self._in_terminal = False
            self._terminal_hold_elapsed = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, face: FaceData, dt: float) -> AttentionState:
        """
        Process one frame.  dt is the elapsed time since the previous frame (s).
        Returns an AttentionState snapshot.
        """
        self._update_face_state(face, dt)
        self._get_distracted_types()
        self._update_awareness(dt)
        self._handle_terminal(dt)

        aw = max(0.0, self._awareness)  # clamp for display

        if self._awareness <= 0.0:
            level = 3
        elif self._awareness <= self._thresh2:
            level = 2
        elif self._awareness <= self._thresh1:
            level = 1
        else:
            level = 0

        calib_pct = min(100, int(self._calib_samples / S.CALIB_FRAMES * 100))

        return AttentionState(
            awareness=aw,
            alert_level=level,
            face_detected=self._face_detected,
            is_distracted=self._driver_distracted,
            distracted_types=dict(self._distracted_types),
            calibrated=self._calib_samples >= S.CALIB_FRAMES,
            calib_percent=calib_pct,
            pitch=self._pitch,
            yaw=self._yaw,
            blink_left=self._blink_left,
            blink_right=self._blink_right,
            terminal_count=self._terminal_count,
            distraction_filter=self._distraction_filter,
        )
