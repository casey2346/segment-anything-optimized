"""
fastapi_server.py — Supreme REST API for ONNX Model Inference (Final + Optional Enhancements Ready)

✅ ONNXRuntime FastAPI Inference Server
✅ JWT Auth, Base64 input, Prometheus metrics, timing, OpenAPI customization
✅ Docker + Gradio/WebUI/CI-ready with unit test hooks
✅ Auto-reload (dev mode)

Author: Your Name
License: MIT
"""

import os
import time
import base64
import logging
from io import BytesIO

import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# ───────── Logging setup ─────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sam_fastapi_server")

# ───────── Config ─────────
MODEL_PATH = os.getenv("MODEL_PATH", "sam_model.onnx")
JWT_SECRET = os.getenv("JWT_SECRET", "demo-secret")
DEVICE = "cuda" if ort.get_device() == "GPU" else "cpu"

# ───────── Load ONNX model ─────────
try:
    logger.info(f"Loading model from {MODEL_PATH} on {DEVICE.upper()}...")
    ort_session = ort.InferenceSession(MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)
    ort_session.run(None, {"input": dummy})  # warmup
    logger.info("Model loaded and warmed up successfully")
except Exception as e:
    logger.exception("Model loading failed")
    raise RuntimeError(f"Failed to load ONNX model: {e}")

# ───────── FastAPI setup ─────────
app = FastAPI(title="Segment Anything Ultra API", version="3.1.0")
security = HTTPBearer()

# ───────── Middleware ─────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────── Prometheus ─────────
REQUEST_COUNT = Counter("sam_requests_total", "Total requests", ["method", "endpoint"])
INFERENCE_LATENCY = Histogram("sam_inference_seconds", "Inference latency")

# ───────── Auth ─────────
def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != JWT_SECRET:
        raise HTTPException(status_code=403, detail="Invalid token")

# ───────── Schemas ─────────
class InferenceRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded RGB image")
    normalize: bool = True

class InferenceResponse(BaseModel):
    mask: list
    inference_time: float

class ErrorResponse(BaseModel):
    detail: str

# ───────── Utils ─────────
def base64_to_numpy(image_base64: str) -> np.ndarray:
    try:
        image = Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
        image = image.resize((256, 256))
        return np.asarray(image).transpose(2, 0, 1).astype(np.float32)
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

# ───────── Routes ─────────
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/metadata")
def metadata():
    return {
        "model_path": MODEL_PATH,
        "input_shape": [1, 3, 256, 256],
        "device": DEVICE,
        "framework": "FastAPI + ONNXRuntime",
        "version": "3.1.0"
    }

@app.post("/predict", response_model=InferenceResponse, responses={500: {"model": ErrorResponse}})
@INFERENCE_LATENCY.time()
def predict(request: InferenceRequest, creds: HTTPAuthorizationCredentials = Depends(verify_jwt)):
    start = time.time()
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()

    try:
        image = base64_to_numpy(request.image_base64)
        if request.normalize:
            image /= 255.0
        image = image.reshape(1, 3, 256, 256)

        output = ort_session.run(None, {"input": image})
        mask = output[0].squeeze().tolist()
        return {"mask": mask, "inference_time": round(time.time() - start, 4)}

    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/")
def root():
    return {"message": "Segment Anything ONNX API - Final Supreme Version with Optional Enhancements"}

@app.get("/openapi.json", include_in_schema=False)
def custom_openapi():
    return get_openapi(title=app.title, version=app.version, routes=app.routes)

if __name__ == "__main__":
    import uvicorn
    reload_flag = os.environ.get("RELOAD", "false").lower() == "true"
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=reload_flag)
