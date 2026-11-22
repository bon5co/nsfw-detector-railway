#!/usr/bin/env python3
"""NSFW image classifier API for Railway.

Two open-weights Vision Transformer classifiers behind one authenticated HTTP API, CPU only:

  * Falconsai/nsfw_image_detection (Apache-2.0 per its model card): ViT-base fine-tuned from
    google/vit-base-patch16-224-in21k on about 80,000 images, labels normal / nsfw, 224x224 input.
    The default model.
  * AdamCodd/vit-base-nsfw-detector (Apache-2.0 per its model card): ViT-base fine-tuned from
    google/vit-base-patch16-384 on about 25,000 images including drawings, labels sfw / nsfw, 384x384
    input. Its card says it is trained to be restrictive (cleavage or a lot of skin scores NSFW) and
    scores 86% accuracy on AI-generated images against 96.5% on its own evaluation set.

What this file adds around them, because neither ships a server:

  1. Bearer API_KEY on every route except GET /health. An empty API_KEY refuses to start.
  2. GET /health answers 503 until both models are loaded and warmed, then 200, unauthenticated.
  3. POST /classify takes one or more images (multipart `file` fields, or one raw image body) and
     returns an NSFW probability per model, plus `flagged` against a threshold.
  4. Upload guards: body size cap, pixel-count cap (decompression bombs), magic-byte format check.
  5. Weights are baked into the image: nothing downloads at boot.
  6. Requests run one at a time behind a lock; threads are sized from the container's cgroup quota.
     Images are held in memory for the request only and are never written to disk or logged.
"""

import io
import os
import socket
import sys
import threading
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

API_KEY = os.environ.get("API_KEY") or ""
THREADS = int(os.environ.get("TORCH_THREADS") or os.cpu_count() or 1)
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES") or 10 * 1024 * 1024)
MAX_FILES = int(os.environ.get("MAX_FILES") or 8)
MAX_PIXELS = int(os.environ.get("MAX_PIXELS") or 50_000_000)
THRESHOLD = float(os.environ.get("NSFW_THRESHOLD") or 0.5)
PUBLIC_PATHS = {"/health"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF"}

MODELS = {
    "falconsai": {"dir": os.environ.get("FALCONSAI_DIR") or "/models/falconsai",
                  "repo": "Falconsai/nsfw_image_detection", "size": 224, "nsfw_label": "nsfw"},
    "adamcodd": {"dir": os.environ.get("ADAMCODD_DIR") or "/models/adamcodd",
                 "repo": "AdamCodd/vit-base-nsfw-detector", "size": 384, "nsfw_label": "nsfw"},
}
DEFAULT_MODEL = "falconsai"

Image.MAX_IMAGE_PIXELS = MAX_PIXELS  # PIL raises DecompressionBombError past 2x this

app = FastAPI(
    title="NSFW image classifier API",
    version="1.0.0",
    description="Falconsai/nsfw_image_detection and AdamCodd/vit-base-nsfw-detector (both Apache-2.0) "
                "on CPU. Authorization: Bearer <API_KEY>.",
)

_state = {"error": None, "loaded_in": None, "ready": False}
_m: dict = {}
_lock = threading.Lock()


def _load() -> None:
    t0 = time.time()
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForImageClassification

        torch.set_num_threads(THREADS)
        for key, spec in MODELS.items():
            model = AutoModelForImageClassification.from_pretrained(spec["dir"]).eval()
            id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
            if spec["nsfw_label"] not in id2label.values():
                raise RuntimeError(f"{key}: no '{spec['nsfw_label']}' label in {id2label}")
            spec["model"] = model
            spec["nsfw_idx"] = next(i for i, l in id2label.items() if l == spec["nsfw_label"])
            spec["labels"] = id2label
        _m.update(torch=torch, np=np)
        warm = Image.new("RGB", (64, 64), (128, 128, 128))
        for key in MODELS:
            _score(warm, key)
        _state["ready"] = True
        _state["loaded_in"] = round(time.time() - t0, 1)
        print(f"[nsfw] ready threads={THREADS} in {_state['loaded_in']}s", flush=True)
    except Exception as exc:  # noqa: BLE001 - reported over /health, never fatal
        _state["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[nsfw] LOAD FAILED: {_state['error']}", flush=True)


def _score(img: Image.Image, key: str) -> float:
    """Probability of the NSFW class. Preprocessing is ViTImageProcessor's: RGB, bilinear resize to a
    square (aspect ratio is not kept), scale to 0-1, normalise with mean 0.5 and std 0.5."""
    np, torch, spec = _m["np"], _m["torch"], MODELS[key]
    size = spec["size"]
    arr = np.asarray(img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    arr = (arr / 255.0 - 0.5) / 0.5
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    with torch.inference_mode():
        logits = spec["model"](pixel_values=tensor).logits
    return float(torch.softmax(logits, dim=-1)[0, spec["nsfw_idx"]])


def _decode(data: bytes, name: str) -> Image.Image:
    if not data:
        raise HTTPException(422, f"{name}: empty image")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"{name}: larger than the {MAX_IMAGE_BYTES} byte limit")
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ALLOWED_FORMATS:
            raise HTTPException(415, f"{name}: {img.format} is not supported, send JPEG, PNG, WebP, GIF, BMP or TIFF")
        if img.width * img.height > MAX_PIXELS:
            raise HTTPException(413, f"{name}: {img.width}x{img.height} exceeds {MAX_PIXELS} pixels")
        img.load()  # first frame only for animated GIF / WebP
        return img
    except HTTPException:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(422, f"{name}: not a readable image ({type(exc).__name__})") from None


@app.middleware("http")
async def guards(request: Request, call_next):
    path = request.url.path.rstrip("/") or "/"
    if path in PUBLIC_PATHS:
        return await call_next(request)
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if not (scheme.lower() == "bearer" and API_KEY and token == API_KEY):
        return JSONResponse({"error": "Missing or invalid Authorization: Bearer <API_KEY>."},
                            status_code=401, headers={"WWW-Authenticate": "Bearer"})
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_IMAGE_BYTES * MAX_FILES + 65536:
        return JSONResponse({"error": "request body is too large"}, status_code=413)
    return await call_next(request)


def _status_kb(field: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return 0


@app.get("/health", include_in_schema=False)
def health():
    if not _state["ready"]:
        return JSONResponse({"status": "loading" if not _state["error"] else "failed",
                             "error": _state["error"]}, status_code=503)
    return {"status": "ok", "loaded_in_s": _state["loaded_in"], "threads": THREADS,
            "rss_mb": _status_kb("VmRSS") // 1024, "peak_rss_mb": _status_kb("VmHWM") // 1024}


@app.get("/v1/models")
def models():
    return {"object": "list", "default": DEFAULT_MODEL, "data": [
        {"id": "falconsai", "repo": MODELS["falconsai"]["repo"], "license": "apache-2.0",
         "input": "224x224", "labels": ["normal", "nsfw"]},
        {"id": "adamcodd", "repo": MODELS["adamcodd"]["repo"], "license": "apache-2.0",
         "input": "384x384", "labels": ["sfw", "nsfw"], "note": "restrictive: swimwear and cleavage score high"}],
        "max_image_bytes": MAX_IMAGE_BYTES, "max_files": MAX_FILES, "default_threshold": THRESHOLD}


@app.post("/classify")
async def classify(request: Request):
    if not _state["ready"]:
        raise HTTPException(503, _state["error"] or "models are still loading, retry in a few seconds")
    qp = request.query_params
    which = qp.get("model", DEFAULT_MODEL)
    keys = list(MODELS) if which == "all" else [which]
    if any(k not in MODELS for k in keys):
        raise HTTPException(422, "`model` must be falconsai, adamcodd or all")
    try:
        threshold = float(qp.get("threshold", THRESHOLD))
    except ValueError:
        raise HTTPException(422, "`threshold` must be a number in (0, 1]") from None
    if not 0 < threshold <= 1:
        raise HTTPException(422, "`threshold` must be a number in (0, 1]")

    ctype = request.headers.get("content-type", "")
    files: list[tuple[str, bytes]] = []
    if ctype.startswith("multipart/form-data"):
        form = await request.form(max_files=MAX_FILES, max_fields=MAX_FILES + 4)
        for field, value in form.multi_items():
            if hasattr(value, "read"):
                files.append((value.filename or field, await value.read(MAX_IMAGE_BYTES + 1)))
    elif ctype.startswith("image/") or ctype == "application/octet-stream":
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_IMAGE_BYTES:
                raise HTTPException(413, f"body: larger than the {MAX_IMAGE_BYTES} byte limit")
        files.append(("body", bytes(body)))
    else:
        raise HTTPException(415, "send multipart/form-data with `file` parts, or a raw image body "
                                 "with an image/* Content-Type")
    if not files:
        raise HTTPException(422, "no image found: attach it as multipart field `file`")
    if len(files) > MAX_FILES:
        raise HTTPException(413, f"at most {MAX_FILES} images per request")
    images = [(name, _decode(data, name), len(data)) for name, data in files]

    t0 = time.time()
    results = []
    with _lock:
        try:
            for name, img, nbytes in images:
                scores = {k: round(_score(img, k), 4) for k in keys}
                top = max(scores.values())
                results.append({"filename": name, "bytes": nbytes, "width": img.width, "height": img.height,
                                "nsfw": scores[keys[0]] if len(keys) == 1 else top,
                                "label": "nsfw" if top >= threshold else "normal",
                                "flagged": bool(top >= threshold),
                                "scores": scores})
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"inference failed: {type(exc).__name__}: {exc}") from exc
    return JSONResponse({"results": results, "threshold": threshold, "models": keys},
                        headers={"X-Processing-Seconds": f"{time.time() - t0:.3f}"})


def main() -> int:
    if not API_KEY:
        print("[nsfw] FATAL: API_KEY is empty; refusing to publish the API to anyone who finds the URL.",
              file=sys.stderr, flush=True)
        return 1
    threading.Thread(target=_load, name="load", daemon=True).start()
    # Railway injects PORT and its healthcheck dials that injected port. asyncio sets
    # IPV6_V6ONLY on a "::" listener, so one socket per family: IPv4 for the edge and
    # healthcheck, IPv6 for the private network (<service>.railway.internal).
    port = int(os.environ.get("PORT") or 8080)
    socks = []
    for family, addr in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((addr, port))
            sock.set_inheritable(True)
            socks.append(sock)
        except OSError as exc:
            print(f"[nsfw] cannot bind {addr}:{port}: {exc}", flush=True)
    if not socks:
        return 1
    uvicorn.Server(uvicorn.Config(app, log_level=os.environ.get("LOG_LEVEL", "warning"))).run(sockets=socks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
