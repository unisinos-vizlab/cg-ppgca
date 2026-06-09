"""
Real-time dashboard for the Driver Attention Monitor.

Layout (1024 × 560 px):
  ┌──────────────────────────────┬───────────────────────────────┐
  │   VIDEO  (700 × 525)         │   STATUS PANEL  (324 × 525)   │
  └──────────────────────────────┴───────────────────────────────┘
  │                  STATUS BAR  (1024 × 35)                      │
  └───────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np

from .face_analyzer import FaceData
from .monitor import AttentionState

# ── Canvas dimensions ──────────────────────────────────────────────────────
CANVAS_W, CANVAS_H = 1024, 560
VIDEO_W, VIDEO_H   = 700, 525
PANEL_X            = VIDEO_W          # right panel starts here
PANEL_W            = CANVAS_W - VIDEO_W
STATUS_BAR_H       = CANVAS_H - VIDEO_H

# ── Colour palette (BGR) ───────────────────────────────────────────────────
C_BG         = (20,  20,  35)
C_PANEL_BG   = (28,  28,  45)
C_TEXT       = (220, 220, 220)
C_TEXT_DIM   = (120, 120, 140)
C_OK         = (60,  200,  80)
C_WARN1      = (30,  210, 220)   # yellow
C_WARN2      = (30,  120, 255)   # orange
C_ALERT      = (40,   40, 220)   # red
C_FACE_MESH  = (80,  160, 255)
C_AXIS_X     = (40,   40, 220)   # BGR red
C_AXIS_Y     = (40,  220,  40)   # BGR green
C_AXIS_Z     = (220,  40,  40)   # BGR blue

# ── MediaPipe face landmark groups for drawing ─────────────────────────────
# These indices trace the face oval, left eye, right eye, lips
_OVAL  = [10,338,297,332,284,251,389,356,454,323,361,288,
          397,365,379,378,400,377,152,148,176,149,150,136,
          172,58,132,93,234,127,162,21,54,103,67,109,10]
_L_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362]
_R_EYE = [33, 7,  163,144,145,153,154,155,133,173,157,158,159,160,161,246,33]


def _alert_color(level: int) -> tuple:
    return [C_OK, C_WARN1, C_WARN2, C_ALERT][level]


def _awareness_color(aw: float) -> tuple:
    """Smooth gradient green → yellow → orange → red."""
    if aw > 0.73:
        return C_OK
    if aw > 0.55:
        t = (aw - 0.55) / (0.73 - 0.55)
        return _lerp_color(C_WARN1, C_OK, t)
    if aw > 0.0:
        t = aw / 0.55
        return _lerp_color(C_ALERT, C_WARN2, t)
    return C_ALERT


def _lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ── Drawing helpers ────────────────────────────────────────────────────────

def _text(img, txt, pos, scale=0.5, color=C_TEXT, thickness=1, bold=False):
    font = cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, txt, pos, font, scale, color, thickness, cv2.LINE_AA)


def _rect(img, pt1, pt2, color, filled=True, radius=6):
    """Rounded-corner rectangle via polyline approximation."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius
    pts = np.array([
        [x1+r, y1], [x2-r, y1], [x2, y1+r], [x2, y2-r],
        [x2-r, y2], [x1+r, y2], [x1, y2-r], [x1, y1+r],
    ])
    if filled:
        cv2.fillPoly(img, [pts], color)
    else:
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)


def _indicator(img, x, y, label, active, color_on=C_OK, color_off=(60,60,80)):
    """Small circle + label indicator."""
    col = color_on if active else color_off
    cv2.circle(img, (x, y), 7, col, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 7, (180,180,180), 1, cv2.LINE_AA)
    _text(img, label, (x + 14, y + 5), scale=0.46, color=C_TEXT if active else C_TEXT_DIM)


def _draw_ring_gauge(img, cx, cy, radius, value: float, level: int) -> None:
    """Circular awareness gauge drawn as a thick arc ring."""
    thickness = 14
    bg_col = (50, 50, 70)
    fg_col = _awareness_color(value)

    # Background ring
    cv2.circle(img, (cx, cy), radius, bg_col, thickness, cv2.LINE_AA)

    # Foreground arc (from -90° clockwise by value * 360°)
    angle_deg = int(value * 360)
    if angle_deg > 0:
        for a in range(0, angle_deg, 2):
            rad = math.radians(-90 + a)
            px = int(cx + radius * math.cos(rad))
            py = int(cy + radius * math.sin(rad))
            cv2.circle(img, (px, py), thickness // 2 + 1, fg_col, -1, cv2.LINE_AA)

    # Center text
    pct_str = f"{int(value * 100)}%"
    tw, th = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)[0]
    _text(img, pct_str, (cx - tw//2, cy + th//2), scale=0.9,
          color=fg_col, thickness=2, bold=True)
    _text(img, "ATENÇÃO", (cx - 34, cy + th//2 + 22), scale=0.42,
          color=C_TEXT_DIM)


def _draw_face_overlay(video: np.ndarray, face: FaceData) -> None:
    """Draw face mesh and head-pose axes on the video frame."""
    if not face.detected or face.landmarks is None:
        return

    lm = face.landmarks
    h, w = video.shape[:2]

    # Face oval outline
    for i in range(len(_OVAL) - 1):
        a, b = _OVAL[i], _OVAL[i+1]
        p1 = (int(lm[a].x * w), int(lm[a].y * h))
        p2 = (int(lm[b].x * w), int(lm[b].y * h))
        cv2.line(video, p1, p2, C_FACE_MESH, 1, cv2.LINE_AA)

    # Eye outlines
    for ring in (_L_EYE, _R_EYE):
        for i in range(len(ring) - 1):
            a, b = ring[i], ring[i+1]
            p1 = (int(lm[a].x * w), int(lm[a].y * h))
            p2 = (int(lm[b].x * w), int(lm[b].y * h))
            cv2.line(video, p1, p2, (255, 220, 80), 1, cv2.LINE_AA)

    # Eye blink indicators (colored dot at eye center)
    for ids, blink in [(_L_EYE, face.blink_left), (_R_EYE, face.blink_right)]:
        xs = [lm[i].x * w for i in ids[:6]]
        ys = [lm[i].y * h for i in ids[:6]]
        cx, cy = int(sum(xs)/len(xs)), int(sum(ys)/len(ys))
        col = C_ALERT if blink > 0.7 else C_OK
        cv2.circle(video, (cx, cy), 5, col, -1, cv2.LINE_AA)

    # Head-pose axes from nose tip
    if face.rvec is not None and face.tvec is not None:
        fl = float(w)
        cam = np.array([[fl,0,w/2],[0,fl,h/2],[0,0,1]], dtype=np.float64)
        dist = np.zeros((4,1))
        axis_len = 60.0
        axis_3d = np.float64([
            [axis_len, 0, 0],
            [0, -axis_len, 0],
            [0, 0, -axis_len],
        ])
        nose_3d = np.float64([[0, 0, 0]])
        all_pts = np.vstack([nose_3d, axis_3d])
        proj, _ = cv2.projectPoints(all_pts, face.rvec, face.tvec, cam, dist)
        n  = tuple(proj[0].ravel().astype(int))
        px = tuple(proj[1].ravel().astype(int))
        py = tuple(proj[2].ravel().astype(int))
        pz = tuple(proj[3].ravel().astype(int))
        cv2.line(video, n, px, C_AXIS_X, 3, cv2.LINE_AA)
        cv2.line(video, n, py, C_AXIS_Y, 3, cv2.LINE_AA)
        cv2.line(video, n, pz, C_AXIS_Z, 2, cv2.LINE_AA)
        cv2.circle(video, n, 4, (255,255,255), -1, cv2.LINE_AA)


# ── Alert messages ─────────────────────────────────────────────────────────
_ALERT_MSGS = [
    "",
    "MANTENHA OS OLHOS NA ESTRADA",
    "ATENÇÃO! VOCÊ ESTÁ DISTRAÍDO",
    "ALERTA MÁXIMO! PARE O VEÍCULO",
]
_ALERT_SUB = [
    "",
    "Nível 1 — Aviso",
    "Nível 2 — Perigo",
    "Nível 3 — TERMINAL",
]


class AttentionDisplay:
    """Renders the attention dashboard onto a fixed-size canvas."""

    def __init__(self) -> None:
        self._blink_toggle = False
        self._last_blink_t = time.monotonic()

    def render(
        self,
        frame: np.ndarray,
        face: FaceData,
        state: AttentionState,
        fps: float = 0.0,
    ) -> np.ndarray:
        """Return a 1024×560 BGR image with video + status panel."""
        canvas = np.full((CANVAS_H, CANVAS_W, 3), C_BG, dtype=np.uint8)
        level = state.alert_level

        # ── 1. Prepare video panel ─────────────────────────────────────────
        video = cv2.resize(frame, (VIDEO_W, VIDEO_H))
        _draw_face_overlay(video, face)
        # Coloured border by alert level
        bdr = _alert_color(level)
        border_px = 4
        video[:border_px, :] = bdr
        video[-border_px:, :] = bdr
        video[:, :border_px] = bdr
        video[:, -border_px:] = bdr

        canvas[:VIDEO_H, :VIDEO_W] = video

        # ── 2. Status panel background ─────────────────────────────────────
        canvas[:VIDEO_H, PANEL_X:] = C_PANEL_BG

        # Vertical divider
        cv2.line(canvas, (PANEL_X, 0), (PANEL_X, VIDEO_H), (60,60,90), 1)

        px0 = PANEL_X + 12   # left text margin inside panel
        py  = 18             # running y cursor

        # Title
        _text(canvas, "MONITOR DE ATENÇÃO", (px0, py), scale=0.52,
              color=C_TEXT, thickness=1, bold=True)
        py += 18
        _text(canvas, "Baseado em OpenPilot DM", (px0, py),
              scale=0.38, color=C_TEXT_DIM)
        py += 20
        cv2.line(canvas, (PANEL_X+6, py), (CANVAS_W-6, py), (60,60,90), 1)
        py += 14

        # ── Awareness gauge ────────────────────────────────────────────────
        gauge_cx = PANEL_X + PANEL_W // 2
        gauge_cy = py + 62
        _draw_ring_gauge(canvas, gauge_cx, gauge_cy, 52, state.awareness, level)
        py = gauge_cy + 70

        cv2.line(canvas, (PANEL_X+6, py), (CANVAS_W-6, py), (60,60,90), 1)
        py += 12

        # ── Distraction indicators ─────────────────────────────────────────
        _text(canvas, "ESTADO DO MOTORISTA", (px0, py), scale=0.41,
              color=C_TEXT_DIM)
        py += 18

        face_ok  = state.face_detected
        pose_ok  = not state.distracted_types.get('pose',  False)
        eye_ok   = not state.distracted_types.get('eye',   False)
        phone_ok = not state.distracted_types.get('phone', False)

        items = [
            ("Face detectada",  face_ok,  C_OK,   C_ALERT),
            ("Pose OK",         pose_ok,  C_OK,   C_WARN2),
            ("Olhos abertos",   eye_ok,   C_OK,   C_WARN2),
            ("Sem celular",     phone_ok, C_OK,   C_WARN2),
        ]
        for label, ok, c_on, c_off in items:
            _indicator(canvas, px0 + 7, py, label, ok,
                       color_on=c_on, color_off=c_off)
            py += 22

        # Blink values under eyes indicator
        _text(canvas,
              f"  EAR: E={face.ear_left:.2f}  D={face.ear_right:.2f}",
              (px0, py), scale=0.38, color=C_TEXT_DIM)
        py += 14

        cv2.line(canvas, (PANEL_X+6, py), (CANVAS_W-6, py), (60,60,90), 1)
        py += 12

        # ── Alert message ──────────────────────────────────────────────────
        if level > 0:
            now = time.monotonic()
            if now - self._last_blink_t > 0.5:
                self._blink_toggle = not self._blink_toggle
                self._last_blink_t = now

            show = level < 3 or self._blink_toggle
            if show:
                col = _alert_color(level)
                _rect(canvas, (PANEL_X+6, py), (CANVAS_W-6, py+50), col)
                msg  = _ALERT_MSGS[level]
                sub  = _ALERT_SUB[level]
                tw   = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
                _text(canvas, msg, (PANEL_X + PANEL_W//2 - tw//2, py+16),
                      scale=0.36, color=(240,240,240), thickness=1)
                tw2  = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
                _text(canvas, sub, (PANEL_X + PANEL_W//2 - tw2//2, py+36),
                      scale=0.42, color=(240,240,240), bold=True)
            py += 56
        else:
            _rect(canvas, (PANEL_X+6, py), (CANVAS_W-6, py+30), C_OK)
            _text(canvas, "MOTORISTA ATENTO", (px0+10, py+20),
                  scale=0.48, color=(20,20,20), thickness=1, bold=True)
            py += 36

        py += 4
        # ── Calibration bar ───────────────────────────────────────────────
        if not state.calibrated:
            bar_w = PANEL_W - 20
            filled = int(bar_w * state.calib_percent / 100)
            cv2.rectangle(canvas, (PANEL_X+10, py), (PANEL_X+10+bar_w, py+10),
                          (60,60,80), -1)
            cv2.rectangle(canvas, (PANEL_X+10, py), (PANEL_X+10+filled, py+10),
                          C_WARN1, -1)
            _text(canvas, f"Calibrando... {state.calib_percent}%",
                  (px0, py+22), scale=0.4, color=C_WARN1)
        else:
            _text(canvas, "✓ Calibrado", (px0, py+14),
                  scale=0.42, color=C_OK)

        # ── 3. Bottom status bar ───────────────────────────────────────────
        bar_y = VIDEO_H
        canvas[bar_y:, :] = (15, 15, 28)
        cv2.line(canvas, (0, bar_y), (CANVAS_W, bar_y), (60,60,90), 1)

        bx = 10
        def _bstat(label, val, x, color=C_TEXT_DIM):
            _text(canvas, f"{label}: {val}", (x, bar_y + 22),
                  scale=0.43, color=color)
            return x + cv2.getTextSize(f"{label}: {val}",
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)[0][0] + 20

        bx = _bstat("Pitch", f"{math.degrees(state.pitch):+.1f}°", bx)
        bx = _bstat("Yaw",   f"{math.degrees(state.yaw):+.1f}°",   bx)
        bx = _bstat("Blink E", f"{face.blink_left:.2f}", bx)
        bx = _bstat("Blink D", f"{face.blink_right:.2f}", bx)
        bx = _bstat("Filtro", f"{state.distraction_filter:.2f}", bx)
        if state.terminal_count > 0:
            bx = _bstat("Alertas terminais", str(state.terminal_count), bx,
                        color=C_ALERT)
        # FPS right-aligned
        fps_str = f"FPS: {fps:.0f}"
        tw = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)[0][0]
        _text(canvas, fps_str, (CANVAS_W - tw - 10, bar_y + 22),
              scale=0.43, color=C_TEXT_DIM)

        # ── 4. Title bar strip at very top of video ────────────────────────
        title = "SISTEMA DE MONITORAMENTO DE ATENÇÃO — OpenPilot DM"
        cv2.rectangle(canvas, (0,0), (VIDEO_W, 22), (0,0,0,160), -1)
        _text(canvas, title, (8, 16), scale=0.44, color=(200,200,200))

        return canvas
