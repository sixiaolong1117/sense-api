FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# 构建时代理（用于 pip 下载加速）
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

CMD ["python3", "app.py"]
