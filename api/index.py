from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse, PlainTextResponse

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "templates" / "index.html"
MODEL_PATH = BASE_DIR / "model_svm.pkl"

_clf = None
_classes: list[str] | None = None


def _load_model():
    global _clf, _classes
    if _clf is not None:
        return
    payload = joblib.load(MODEL_PATH)
    _clf = payload["model"]
    _classes = list(payload["classes"])


class PredictRequest(BaseModel):
    vector: list[float] = Field(..., min_length=63, max_length=63)


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


@app.post("/predict")
async def predict(req: PredictRequest):
    if not MODEL_PATH.is_file():
        return {"ok": False, "error": "Missing model_svm.pkl on server."}

    _load_model()
    x = np.asarray(req.vector, dtype=np.float32).reshape(1, -1)
    proba = _clf.predict_proba(x)[0]
    top = int(np.argmax(proba))
    confidence = float(proba[top])
    label = str(_classes[top])
    return {"ok": True, "label": label, "confidence": confidence}

