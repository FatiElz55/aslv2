import base64
import os
import threading
import urllib.request
from pathlib import Path

import cv2
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from hand_detector import HandDetector

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "templates" / "index.html"
MODEL_PATH = BASE_DIR / "model_svm.pkl"
FALLBACK_MODEL_PATH = BASE_DIR.parent / "model_svm.pkl"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONFIDENCE_THRESH = 0.40
PREVIEW_JPEG_QUALITY = 70

_model_lock = threading.Lock()
_clf = None
_classes: list[str] | None = None

_detect_lock = threading.Lock()
_detector: HandDetector | None = None


def _maybe_download_model(dst_path: Path) -> bool:
    url = os.environ.get("MODEL_URL", "").strip()
    if not url:
        return False
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading model to {dst_path} …")
        urllib.request.urlretrieve(url, str(dst_path))
        print("Model download complete.")
        return True
    except Exception as exc:
        print("Model download failed:", exc)
        return False


def _ensure_loaded():
    global _clf, _classes, _detector
    if _clf is not None and _detector is not None:
        return

    model_path = MODEL_PATH if MODEL_PATH.is_file() else FALLBACK_MODEL_PATH
    if not model_path.is_file():
        if _maybe_download_model(MODEL_PATH):
            model_path = MODEL_PATH

    if not model_path.is_file():
        raise RuntimeError(
            "Missing model_svm.pkl on server. Put it inside `asl/` or set MODEL_URL."
        )

    payload = joblib.load(model_path)
    _clf = payload["model"]
    _classes = list(payload["classes"])
    _detector = HandDetector(
        max_hands=1,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
        min_tracking_confidence=0.25,
    )


class PredictVectorRequest(BaseModel):
    vector: list[float] = Field(..., min_length=63, max_length=63)


class PredictFrameRequest(BaseModel):
    image: str = Field(..., min_length=10)


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/robots.txt")
async def robots():
    return PlainTextResponse("User-agent: *\nDisallow:\n")


@app.get("/", response_class=HTMLResponse)
async def index():
    if not INDEX_PATH.is_file():
        return HTMLResponse(
            "<h1>ASL Alphabet</h1><p>Missing <code>asl/templates/index.html</code></p>",
            status_code=500,
        )
    return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))


def _predict_from_vector(vec: np.ndarray):
    _ensure_loaded()
    with _model_lock:
        proba = _clf.predict_proba(vec.reshape(1, -1))[0]
        top = int(np.argmax(proba))
        confidence = float(proba[top])
        label = str(_classes[top])
    if confidence < CONFIDENCE_THRESH:
        return "", confidence
    return label, confidence


def _encode_preview_jpeg(bgr: np.ndarray) -> str:
    small = cv2.resize(bgr, (620, 465), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(
        ".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_JPEG_QUALITY]
    )
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


@app.post("/predict")
async def predict(req: PredictVectorRequest):
    try:
        x = np.asarray(req.vector, dtype=np.float32)
        if x.shape != (63,):
            return {"ok": False, "error": "vector must be length 63"}
        label, confidence = _predict_from_vector(x)
        return {"ok": True, "label": label, "confidence": confidence}
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/predict_frame")
async def predict_frame(req: PredictFrameRequest):
    try:
        _ensure_loaded()
        raw = base64.b64decode(req.image)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"ok": False, "error": "Could not decode image"}
        if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        with _detect_lock:
            vector, annotated = _detector.get_landmarks(frame)

        if vector is None:
            return {"ok": True, "label": "", "confidence": 0.0, "preview": ""}

        label, confidence = _predict_from_vector(vector)
        preview = _encode_preview_jpeg(annotated)
        return {"ok": True, "label": label, "confidence": confidence, "preview": preview}
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}

