# nsfw-detector-railway

NSFW image classifier API for [Railway](https://railway.com), CPU only. Two open-weights Vision Transformer classifiers behind one authenticated HTTP API:

- [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) (Apache-2.0 per its model card): normal / nsfw, 224x224 input, the default.
- [AdamCodd/vit-base-nsfw-detector](https://huggingface.co/AdamCodd/vit-base-nsfw-detector) (Apache-2.0 per its model card): sfw / nsfw, 384x384 input, trained to be restrictive (swimwear and cleavage score high), includes drawings.

- `POST /classify` with multipart `file` parts (up to 8) or one raw image body with an `image/*` Content-Type. Query: `model=falconsai|adamcodd|all`, `threshold=0.5`. Returns `{"results": [{"filename", "width", "height", "nsfw", "label", "flagged", "scores": {...}}]}`.
- `GET /health` (503 until both models are loaded and warm, then 200), `GET /v1/models`.
- Auth: `Authorization: Bearer $API_KEY` on every route except `/health`. An empty `API_KEY` refuses to start.
- Env: `API_KEY` (required), `NSFW_THRESHOLD` (0.5), `MAX_IMAGE_BYTES` (10485760), `MAX_FILES` (8), `MAX_PIXELS` (50000000), `TORCH_THREADS` (from the cgroup quota).

Weights are baked into the image, pinned to one Hugging Face commit each and verified by sha256 at build. Images are held in memory for one request and never logged or stored. Wrapper code is Apache-2.0 (`LICENSE`); see `NOTICE`.

Image: `ghcr.io/bon5co/nsfw-detector-railway`.
