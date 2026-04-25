# ASL Alphabet (Vercel deploy copy)

This `asl/` folder is a **deployable copy** of the project intended for hosting (ex: **Vercel**).

It intentionally **does not** include: 
- datasets
- the trained model file (`model_svm.pkl`) (you must provide it for hosting)

## Deploy to Vercel

In Vercel, set the **Root Directory** to `asl/`.

Endpoints:
- `GET /` (SignWriter UI)
- `GET /health`
- `POST /predict` (send 63-number hand-landmark vector; server runs SVM)

## Run the full realtime app locally

Use the repository root (not `asl/`):

```bash
pip install -r requirements.txt
python app.py
```

## Provide the trained model for hosting

The deployed backend loads `asl/model_svm.pkl`.

Options:
- Put **`model_svm.pkl` inside `asl/`** before deploying.
- Or set environment variable **`MODEL_URL`** to a direct-download URL of the file; the server will download it at runtime.

## Note about deploy size (Vercel)

To stay under Vercel’s serverless size limits, this deploy copy runs **MediaPipe in the browser** (JS),
and sends only the 63-number vector to the backend. This keeps the backend lightweight and deployable.
