"""HoloPuck computer-vision helpers.

* TableView   - perspective (homography) mapping between the flat game table
                and the camera frame, plus alpha-blended AR overlay.
* HandTracker - MediaPipe HandLandmarker wrapper returning the index fingertip.
* Smoother    - exponential moving average to steady a jittery fingertip.
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from game import TABLE_H, TABLE_W

INDEX_FINGER_TIP = 8  # MediaPipe hand landmark id (0 = wrist ... 8 = index fingertip)
OVERLAY_ALPHA = 0.80  # how opaque the virtual table is over the camera image

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

# The 21 bones of the MediaPipe hand skeleton (pairs of landmark ids), used for drawing only.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # little finger
    (0, 17),                                  # edge of the palm
]

# Corners of the table inside the camera frame, as fractions of (width, height),
# in the order: top-left, top-right, bottom-right, bottom-left.
QUAD_PERSPECTIVE = [(0.20, 0.18), (0.80, 0.18), (0.94, 0.93), (0.06, 0.93)]
QUAD_FLAT = [(0.10, 0.15), (0.90, 0.15), (0.90, 0.95), (0.10, 0.95)]


class TableView:
    """Maps between table coordinates and camera-frame pixels with a homography."""

    def __init__(self, frame_w: int, frame_h: int, perspective: bool = True):
        self.fw, self.fh = frame_w, frame_h
        self.set_perspective(perspective)

    def set_perspective(self, on: bool) -> None:
        self.perspective = on
        src = np.float32([[0, 0], [TABLE_W, 0], [TABLE_W, TABLE_H], [0, TABLE_H]])
        quad = QUAD_PERSPECTIVE if on else QUAD_FLAT
        dst = np.float32([[x * self.fw, y * self.fh] for x, y in quad])
        self.M = cv2.getPerspectiveTransform(src, dst)       # table  -> camera
        self.M_inv = cv2.getPerspectiveTransform(dst, src)   # camera -> table
        mask = cv2.warpPerspective(
            np.full((TABLE_H, TABLE_W), 255, np.uint8), self.M, (self.fw, self.fh)
        )
        self._alpha = (mask.astype(np.float32) / 255.0 * OVERLAY_ALPHA)[..., None]

    def frame_to_table(self, x: float, y: float) -> tuple[float, float]:
        pt = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), self.M_inv)
        return float(pt[0, 0, 0]), float(pt[0, 0, 1])

    def table_to_frame(self, x: float, y: float) -> tuple[float, float]:
        pt = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), self.M)
        return float(pt[0, 0, 0]), float(pt[0, 0, 1])

    def overlay(self, frame: np.ndarray, table_canvas: np.ndarray) -> np.ndarray:
        """Warp the rendered table into the camera view and alpha-blend it."""
        warped = cv2.warpPerspective(table_canvas, self.M, (self.fw, self.fh))
        out = frame.astype(np.float32) * (1.0 - self._alpha) + warped.astype(np.float32) * self._alpha
        return out.astype(np.uint8)


class Smoother:
    """Exponential moving average: s = a*new + (1-a)*s.  Larger alpha = less lag."""

    def __init__(self, alpha: float = 0.55):
        self.alpha = alpha
        self.value: np.ndarray | None = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        new = np.array([x, y], dtype=float)
        self.value = new if self.value is None else self.alpha * new + (1 - self.alpha) * self.value
        return float(self.value[0]), float(self.value[1])

    def reset(self) -> None:
        self.value = None


def ensure_model(path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Return the hand-landmark model file, downloading it once if it is missing."""
    path = Path(path)
    if path.exists():
        return path
    print(f"Downloading hand tracking model (one time, ~8 MB) to {path} ...")
    tmp = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(path)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"Could not download the hand tracking model ({exc}).\n"
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it as:\n  {path}\n"
            "or run with --mouse to play without the camera."
        ) from exc
    return path


class HandTracker:
    """Finds one hand with MediaPipe HandLandmarker; returns the index fingertip in pixels."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH,
                 detection_conf: float = 0.6, tracking_conf: float = 0.5):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("MediaPipe is not installed. Run: pip install -r requirements.txt") from exc

        self._mp = mp
        model = ensure_model(Path(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,  # tracks across frames (faster, steadier)
            num_hands=1,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=tracking_conf,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._connections = HAND_CONNECTIONS
        self._landmarks = None
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray) -> tuple[float, float] | None:
        """Return (x, y) of the index fingertip in frame pixels, or None if no hand."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # MediaPipe expects RGB, OpenCV gives BGR
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)  # timestamps must increase
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        if not result.hand_landmarks:
            self._landmarks = None
            return None
        self._landmarks = result.hand_landmarks[0]  # 21 landmarks, coordinates normalised to 0..1
        h, w = frame_bgr.shape[:2]
        tip = self._landmarks[INDEX_FINGER_TIP]
        return tip.x * w, tip.y * h

    def draw(self, frame_bgr: np.ndarray) -> None:
        """Draw the 21 hand landmarks and the hand skeleton on the frame (in place)."""
        if self._landmarks is None:
            return
        h, w = frame_bgr.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in self._landmarks]
        for a, b in self._connections:
            cv2.line(frame_bgr, pts[a], pts[b], (80, 220, 80), 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            cv2.circle(frame_bgr, p, 6 if i == INDEX_FINGER_TIP else 3,
                       (0, 0, 255) if i == INDEX_FINGER_TIP else (255, 255, 255), -1, cv2.LINE_AA)

    def close(self) -> None:
        self._landmarker.close()


class SkinTracker:
    """Classical image-processing hand tracker (no machine learning, no model download).

    Pipeline:  Gaussian blur -> BGR to YCrCb -> skin-colour threshold -> morphological
    open/close -> largest contour -> topmost point = fingertip.

    It has the same interface as HandTracker (process / draw / close), so main.py can
    use either one.  Tips: use a plain background, keep your face out of the lower part
    of the frame, and point one finger upwards.
    """

    # Skin range in YCrCb (Y is brightness, Cr/Cb are colour) - works for many skin tones
    SKIN_LO = (0, 133, 77)
    SKIN_HI = (255, 173, 127)

    def __init__(self, roi_top: float = 0.20, min_area_frac: float = 0.004):
        self.roi_top = roi_top              # ignore this top fraction of the frame (face/ceiling)
        self.min_area_frac = min_area_frac  # ignore blobs smaller than this fraction of the frame
        self.mask: np.ndarray | None = None  # last binary mask (for the debug inset)
        self._contour: np.ndarray | None = None
        self._tip: tuple[int, int] | None = None

    def process(self, frame_bgr: np.ndarray) -> tuple[float, float] | None:
        """Return (x, y) of the fingertip in frame pixels, or None if no hand-sized skin blob."""
        h, w = frame_bgr.shape[:2]
        blur = cv2.GaussianBlur(frame_bgr, (7, 7), 0)              # 1. smooth away sensor noise
        ycrcb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)             # 2. colour space where skin clusters
        mask = cv2.inRange(ycrcb, np.array(self.SKIN_LO), np.array(self.SKIN_HI))  # 3. threshold
        mask[: int(h * self.roi_top), :] = 0                        # region of interest
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)    # 4. remove speckles
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)   #    fill small holes
        self.mask = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 5. shapes
        self._contour, self._tip = None, None
        if not contours:
            return None
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) < self.min_area_frac * w * h:
            return None
        self._contour = biggest

        pts = biggest[:, 0, :]                       # 6. extreme point: the topmost contour pixels
        top_y = int(pts[:, 1].min())
        band = pts[pts[:, 1] <= top_y + 6]           # average a thin band so the tip is steady
        self._tip = (int(band[:, 0].mean()), top_y)
        return float(self._tip[0]), float(self._tip[1])

    def draw(self, frame_bgr: np.ndarray) -> None:
        """Draw the hand outline, its convex hull and the fingertip (in place)."""
        if self._contour is None:
            return
        cv2.drawContours(frame_bgr, [self._contour], -1, (80, 220, 80), 2, cv2.LINE_AA)
        hull = cv2.convexHull(self._contour)
        cv2.drawContours(frame_bgr, [hull], -1, (0, 200, 255), 1, cv2.LINE_AA)
        if self._tip is not None:
            cv2.circle(frame_bgr, self._tip, 8, (0, 0, 255), -1, cv2.LINE_AA)

    def close(self) -> None:
        pass
