# Whisper ASR + Qwen2-VL HTTP API

基于 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 和 [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) 的多模态 AI API 服务，使用 CUDA GPU 加速，通过 Docker 一键部署。

- **语音识别** — 上传音频文件，返回转录文本
- **视觉识别** — 上传图片文件，返回图文理解结果

## 项目结构

```
whisper/
├── app.py              # FastAPI 应用，提供 /transcribe 和 /vision 接口
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

首次启动会自动下载 Whisper 模型（约 3GB）和 Qwen2-VL-2B 模型（约 4GB），模型文件持久化在 `./models` 目录，后续启动无需重复下载。

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

## 语音识别 API

### 接口

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

### 支持的音频格式

只要 FFmpeg 能解码的格式都支持，包括但不限于：

- WAV / MP3 / FLAC / M4A / AAC / OGG / WMA

---

## 视觉识别 API

### 接口

```
POST /vision
```

### 请求格式

`multipart/form-data`，字段名 `file` 传入图片，可选字段 `prompt` 传入文本提示词。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | 是 | 图片文件（JPG / PNG / WEBP / BMP 等） |
| `prompt` | string | 否 | 文本提示词，默认 `"请描述这张图片的内容"` |

### curl 调用

```bash
curl -F "file=@photo.jpg" http://localhost:56178/vision

# 自定义提示词
curl -F "file=@chart.png" -F "prompt=请分析这张图表并总结关键信息" http://localhost:56178/vision
```

### Python 调用

```python
import requests

files = {"file": open("photo.jpg", "rb")}
r = requests.post("http://192.168.1.100:56178/vision", files=files)
print(r.json())

# 自定义提示词
files = {"file": open("chart.png", "rb")}
data = {"prompt": "请分析这张图表"}
r = requests.post("http://192.168.1.100:56178/vision", files=files, data=data)
print(r.json())
```

### 返回格式

```json
{
  "text": "图片中是一个现代化的办公空间，有白色的桌子和灰色椅子..."
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | string | 视觉识别结果文本 |

### 支持的图片格式

- JPG / JPEG / PNG / WEBP / BMP / GIF

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
    shm_size: '2gb'
    ports:
      - "56178:56178"
    volumes:
      - ./models:/root/.cache/huggingface
    restart: unless-stopped
    gpus: all
```

> `shm_size: '2gb'` 是 Qwen2-VL 的必要配置，用于 DataLoader 的共享内存，请勿移除。

## 资源管理

### 模型闲置自动释放显存

所有模型均使用 `ModelManager` 统一管理，在闲置 **5 分钟** 后自动卸载并清理 GPU 显存，下次请求时自动重新加载。可通过 `app.py` 中的 `idle_timeout` 参数调整超时阈值。

## 自定义模型

### Whisper 模型

编辑 `app.py` 中的 `_load_whisper()` 函数：

```python
def _load_whisper():
    return WhisperModel(
        "large-v3-turbo",           # 模型名称
        device="cuda",              # 推理设备: cuda / cpu
        compute_type="float16"      # 精度: float16 / int8_float16 / float32
    )
```

### Qwen2-VL 模型

编辑 `app.py` 中的 `QWEN_MODEL_NAME` 变量：

```python
QWEN_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"  # 可选 7B/72B
```

> **注意**：切换到 CPU 推理请移除 GPU 相关配置，Qwen2-VL 在 CPU 上速度较慢，建议使用 GPU。

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

**Q: 显存不足怎么办？**

A: 两个模型不会同时占用显存 — 闲置 5 分钟会自动卸载。如果一张 GPU 显存仍然紧张（如 6GB），可考虑：
1. 缩短 `idle_timeout` 让模型更快释放
2. 使用更小的模型（如 `Qwen/Qwen2-VL-2B-Instruct` 已是最小）
3. 切换到 `compute_type="int8_float16"` 降低 Whisper 显存占用
