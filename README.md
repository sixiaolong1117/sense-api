# Whisper ASR HTTP API

基于 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 的语音识别 HTTP API 服务，使用 CUDA GPU 加速，通过 Docker 一键部署。

## 项目结构

```
whisper/
├── app.py              # FastAPI 应用，提供 /transcribe 接口
├── Dockerfile          # 基于 CUDA 12.8 的 Docker 镜像构建
├── docker-compose.yml  # 容器编排，含 GPU 资源配置
├── requirements.txt    # Python 依赖
└── README.md           # 本文件
```

## 环境要求

- **Docker** 24+ 及 Docker Compose
- **NVIDIA GPU**（支持 CUDA 12.8）
- **NVIDIA Container Toolkit**（[安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)）

## 快速启动

```bash
# 构建并启动
docker compose up -d

# 查看日志
docker compose logs -f
```

首次启动会自动下载 `large-v3` 模型（约 3GB），模型文件持久化在 `./models` 目录，后续启动无需重复下载。

### 停止并清理

```bash
# 停止容器
docker compose down

# 停止容器并同时删除镜像（下次 up 会重新构建）
docker compose down --rmi all
```

### 重新构建

修改代码或依赖后，需要重新构建镜像并启动：

```bash
docker compose up -d --build
```

## API 用法

### 接口地址

```
POST /transcribe
```

### 请求格式

`multipart/form-data`，字段名 `file`，传入音频文件。

### curl 调用

```bash
curl -F "file=@test.wav" http://localhost:56178/transcribe
```

### Python 调用

```python
import requests

files = {"file": open("audio.wav", "rb")}
r = requests.post("http://192.168.1.100:56178/transcribe", files=files)
print(r.json())
```

### 返回格式

```json
{
  "language": "zh",
  "text": "你好，欢迎使用 Whisper。"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `language` | string | 检测到的音频语言（ISO 639-1 代码） |
| `text` | string | 识别出的文本 |

## 支持的音频格式

只要 FFmpeg 能解码的格式都支持，包括但不限于：

- WAV / MP3 / FLAC / M4A / AAC / OGG / WMA

## Docker Compose 配置说明

### 默认配置（推荐）

使用 `deploy.resources.devices` 声明 GPU 资源（适用于 Docker Swarm 及较新版本）：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

### 备选配置

如果上述方式报错，可改用 `gpus` 字段：

```yaml
services:
  whisper:
    build: .
    ports:
      - "56178:56178"
    volumes:
      - ./models:/root/.cache/huggingface
    restart: unless-stopped
    gpus: all
```

## 自定义模型

编辑 `app.py`，修改模型名称或参数：

```python
model = WhisperModel(
    "large-v3",          # 模型名称，可选: tiny/base/small/medium/large-v3
    device="cuda",       # 推理设备: cuda / cpu
    compute_type="float16"  # 精度: float16 / int8_float16 / float32
)
```

> **注意**：切换到 CPU 推理请删除 `compute_type` 参数或设为 `int8`，并移除 Docker Compose 中的 GPU 配置。

## 常见问题

**Q: 启动后日志显示模型下载失败？**

A: 确保网络能访问 HuggingFace。可通过设置 `HF_ENDPOINT` 环境变量使用镜像：

```yaml
environment:
  - HF_ENDPOINT=https://hf-mirror.com
```

**Q: 提示 "CUDA error: no kernel image is available"？**

A: 请确认 GPU 驱动版本 ≥ 550，且支持 CUDA 12.x。

**Q: 如何指定使用某一张 GPU？**

A: 使用 `CUDA_VISIBLE_DEVICES` 环境变量：

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0   # 仅使用第 1 张 GPU
```
