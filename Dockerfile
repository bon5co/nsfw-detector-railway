# NSFW image classifier API: two ViT classifiers (Falconsai + AdamCodd), Bearer API_KEY.
# Source: https://github.com/bon5co/nsfw-detector-railway
# CPU only (torch), weights baked in (about 690 MB), stateless (no volume needed).
# upstream: pytorch/pytorch torch 2.15.0 and huggingface/transformers 5.24.0 (newest releases, checked 2026-11-22) are pinned in requirements.txt
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
    && pip list 2>/dev/null | grep -iE "^(torch|transformers|pillow|safetensors|numpy) "

# Weights, each pinned to one Hugging Face commit and verified by sha256 so a changed or replaced file
# fails the build instead of shipping. A download at boot would sit inside Railway's healthcheck window.
#   Falconsai/nsfw_image_detection   Apache-2.0 (model card metadata; the repository ships no LICENSE file)
#   AdamCodd/vit-base-nsfw-detector  Apache-2.0 (model card metadata; the repository ships no LICENSE file)
ARG HF_FAL=https://huggingface.co/Falconsai/nsfw_image_detection/resolve/96cb0d0342c7afb80cab76ecc58b265fa44da256
ARG HF_ADAM=https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/8587de998f441aac03fdd57a85d2e4cb808c7d64
ADD --checksum=sha256:cd8c0a19566f199b5c64da402f1d6a9daf80d094fe6536befb060a2cc59449c2 ${HF_FAL}/config.json /models/falconsai/config.json
ADD --checksum=sha256:97b2ce64ec146884b37f98ee7944ca4891aa72f6827dc0cb10684a1cbecd5830 ${HF_FAL}/model.safetensors /models/falconsai/model.safetensors
ADD --checksum=sha256:2e6e2cd12fbe7d70e992ef649eae85e6d72d87c9cfc31b4cecd36de0d5facadc ${HF_FAL}/README.md /models/falconsai/MODEL_CARD.md
ADD --checksum=sha256:604d1da5e3cc8b6dc3f2768a911ed61a44a331cfcee9ecde2ce2cba9a1e7972b ${HF_ADAM}/config.json /models/adamcodd/config.json
ADD --checksum=sha256:266efb8bf67c1e865a577222fbbd6ddb149b9e00ba0d2b50466a034837f026a4 ${HF_ADAM}/model.safetensors /models/adamcodd/model.safetensors
ADD --checksum=sha256:325e88a63de1d94f1f31dcf69b07e8e302723ab8fc5ae2c2f9cdd7963acc753f ${HF_ADAM}/README.md /models/adamcodd/MODEL_CARD.md

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin nsfw \
    && chmod -R a+rX /models
WORKDIR /app
COPY app.py entrypoint.sh NOTICE LICENSE /app/
RUN chmod +x /app/entrypoint.sh
ENV FALCONSAI_DIR=/models/falconsai ADAMCODD_DIR=/models/adamcodd
USER 1000
EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "/app/app.py"]
