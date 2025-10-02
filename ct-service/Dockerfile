# CUDA runtime + PyTorch
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# Системные зависимости для opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models
RUN mkdir -p reports tmp

ENV K_SLICES=5 IMG_SIZE=320 UNCERT_LOW=0.45 UNCERT_HIGH=0.55 DEVICE=cuda

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host","0.0.0.0", "--port","8000"]
