# Sense API — 开发指南

## 项目概述

单文件 FastAPI 服务 (`app.py`)，提供三个接口：
- `GET /health` — 健康检查
- `POST /transcribe` — faster-whisper 语音识别
- `POST /vision` — Qwen2-VL-2B 图片理解

运行在端口 **56178**，依赖 NVIDIA GPU (CUDA)，上传文件上限 **50MB**。

## 关键约束

- **`shm_size: '2gb'` 必须保留** — Qwen2-VL 的 DataLoader 需要共享内存，删掉会崩溃。
- **模型闲置 5 分钟自动卸载** — `ModelManager` 在后台线程监控，超时后释放 GPU 显存和 CPU 内存。两个模型不会同时占用显存。
- **`docker-compose.yml` 被 gitignore** — 基于 `docker-compose.yml.example` 复制并修改，不要直接提交。
- **模型权重在 `./models` 目录** — 通过 volume 挂载到容器内 `/root/.cache/huggingface`，首次启动自动下载。

## 本地开发

项目没有本地开发服务器配置。修改代码后通过 Docker 验证：

```bash
docker compose up -d --build   # 重新构建并启动
docker compose logs -f         # 查看日志
```

## 测试 API

```bash
# 健康检查
curl http://localhost:56178/health

# 语音识别
curl -F "file=@test.wav" http://localhost:56178/transcribe

# 视觉识别
curl -F "file=@photo.jpg" http://localhost:56178/vision
curl -F "file=@chart.png" -F "prompt=请分析这张图表" http://localhost:56178/vision
```

## CI/CD

- Push 到 `main` → 自动构建 Docker 镜像，打 `dev` 标签推送到 GHCR
- 手动触发 → 可选 `dev` 或 `latest` 标签
- 镜像地址: `ghcr.io/sixiaolong1117/sense-api`

## 环境变量

复制 `.env.example` 为 `.env`：
- `HF_ENDPOINT` — HuggingFace 镜像站（国内加速用 `https://hf-mirror.com`）
- `HTTP_PROXY` / `HTTPS_PROXY` — 构建时代理（pip 下载）和运行时代理

`docker-compose.yml.example` 的 `build.args` 会自动从 `.env` 读取代理配置传入 Dockerfile。

## 模型管理

`ModelManager` 类（`app.py:36`）统一管理模型生命周期：
- 每个模型独立线程锁和监控
- 卸载时执行完整清理：`torch.cuda.empty_cache()` → `gc.collect()` → `_trim_memory()`
- `_trim_memory()` 仅在 Linux 下调用 `malloc_trim(0)` 归还 CPU 内存
- 调用 `get_model()` 时若模型未加载会自动下载并加载
