import os
import urllib.request
from dataclasses import replace

import cv2
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
)
from mediapipe.tasks.python.vision.core.image import Image as MPImage
from mediapipe.tasks.python.vision.core.image import ImageFormat

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "mediapipe", "hand_landmarker.task"
)


def _ensure_hand_model(path: str = _MODEL_PATH) -> str:
    if not os.path.isfile(path):
        print(f"Downloading MediaPipe hand model to {path} …")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, path)
        print("Download complete.")
    return path


class HandDetector:
    """Hand landmarks via MediaPipe Tasks (compatible with Python 3.13)."""

    def __init__(
        self,
        max_hands: int = 1,
        min_hand_detection_confidence: float = 0.25,
        min_hand_presence_confidence: float = 0.25,
        min_tracking_confidence: float = 0.25,
    ):
        model_path = _ensure_hand_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=max_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._connections = HandLandmarksConnections.HAND_CONNECTIONS

    def _draw_hands(self, frame: np.ndarray, landmarks_list: list) -> None:
        h, w = frame.shape[:2]
        for hand in landmarks_list:
            for conn in self._connections:
                a, b = hand[conn.start], hand[conn.end]
                if a.x is None or b.x is None:
                    continue
                cv2.line(
                    frame,
                    (int(a.x * w), int(a.y * h)),
                    (int(b.x * w), int(b.y * h)),
                    (0, 255, 0),
                    2,
                )
            for lm in hand:
                if lm.x is None:
                    continue
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 0, 255), -1)

    def get_landmarks(self, frame: np.ndarray):
        """Return (feature_vector, annotated_frame) or (None, frame) if no hand found.

        feature_vector is a float32 ndarray of shape (63,): 21 landmarks × (x, y, z),
        centred on the wrist and normalised by the maximum absolute coordinate value.
        """
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        def _detect(arr):
            return self._landmarker.detect(MPImage(ImageFormat.SRGB, arr))

        result = _detect(rgb)
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
        else:
            result_m = _detect(np.ascontiguousarray(cv2.flip(rgb, 1)))
            if not result_m.hand_landmarks:
                return None, frame
            hand = [
                replace(lm, x=(1.0 - lm.x) if lm.x is not None else None)
                for lm in result_m.hand_landmarks[0]
            ]

        self._draw_hands(frame, [hand])

        raw = [(lm.x, lm.y, lm.z if lm.z is not None else 0.0) for lm in hand]
        wrist_x, wrist_y, wrist_z = raw[0]
        normalised = [(x - wrist_x, y - wrist_y, z - wrist_z) for x, y, z in raw]

        vector = np.array(normalised, dtype=np.float32).flatten()
        max_val = np.max(np.abs(vector)) + 1e-6
        vector /= max_val

        return vector, frame
