# kanvis-segmentation — standalone SAM3 / ROI gRPC service.
#   docker build --target gpu -t kanvis-segmentation:gpu .
#   docker build --target cpu -t kanvis-segmentation:cpu .

FROM python:3.12-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    KANVIS_SEGMENTATION_LISTEN=0.0.0.0:50100 \
    KANVIS_SEGMENTATION_METRICS_PORT=9109 \
    SEGMENTATION_WORK_DIR=/tmp/kanvis-segmentation
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
COPY requirements-service.txt /app/requirements-service.txt
RUN pip install --no-cache-dir -r /app/requirements-service.txt

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04 AS gpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    KANVIS_SEGMENTATION_LISTEN=0.0.0.0:50100 \
    KANVIS_SEGMENTATION_METRICS_PORT=9109 \
    SEGMENTATION_WORK_DIR=/tmp/kanvis-segmentation \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip git ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python
COPY requirements.txt requirements-service.txt /app/
RUN python3 -m pip install --no-cache-dir --break-system-packages \
      torch --index-url https://download.pytorch.org/whl/cu128 && \
    python3 -m pip install --no-cache-dir --break-system-packages \
      -r /app/requirements-service.txt -r /app/requirements.txt
# Ultralytics pulls a PyPI torchvision without the CUDA NMS op.
RUN python3 -m pip uninstall -y --break-system-packages torchvision && \
    python3 -m pip install --no-cache-dir --break-system-packages \
      torchvision --index-url https://download.pytorch.org/whl/cu128
COPY proto /app/proto
COPY kanvis /app/kanvis
COPY kanvis_segmentation /app/kanvis_segmentation
COPY detector_palma /app/detector_palma
COPY pipeline_manos.py sam3_mano_objeto.py salida_comun.py /app/
COPY identifiers.yaml identificadores.yaml /app/
COPY tools /app/tools
COPY scripts /app/scripts
COPY pesos /app/pesos
COPY fixtures /app/fixtures
RUN python3 -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. \
      proto/kanvis/segmentation/v1/segmentation.proto
# MediaPipe Holistic needs EGL at import/load time.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libegl1 libgles2 libglx-mesa0 libopengl0 && \
    rm -rf /var/lib/apt/lists/*
EXPOSE 50100 9109
CMD ["python3", "-m", "kanvis_segmentation.server"]

FROM base AS cpu
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip uninstall -y torchvision && \
    pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cpu
COPY proto /app/proto
COPY kanvis /app/kanvis
COPY kanvis_segmentation /app/kanvis_segmentation
COPY detector_palma /app/detector_palma
COPY pipeline_manos.py sam3_mano_objeto.py salida_comun.py /app/
COPY identifiers.yaml identificadores.yaml /app/
COPY tools /app/tools
COPY scripts /app/scripts
COPY fixtures /app/fixtures
RUN python3 -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. \
      proto/kanvis/segmentation/v1/segmentation.proto
RUN apt-get update && apt-get install -y --no-install-recommends \
      libegl1 libgles2 libglx-mesa0 libopengl0 && \
    rm -rf /var/lib/apt/lists/*
EXPOSE 50100 9109
CMD ["python3", "-m", "kanvis_segmentation.server"]
