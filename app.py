from fastapi import FastAPI, UploadFile
from faster_whisper import WhisperModel
import tempfile
import time
import threading
import torch

app = FastAPI()

# ============ 闲置自动释放显存机制 ============
MODEL_NAME = "large-v3-turbo"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
IDLE_TIMEOUT = 300  # 闲置超时（秒），5 分钟无请求则卸载模型

model_lock = threading.Lock()
model = None
last_request_time = time.time()


def _load_model():
    """加载模型（调用前必须先持有 model_lock）"""
    global model
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)


def _unload_model():
    """卸载模型并释放 GPU 显存（调用前必须先持有 model_lock）"""
    global model
    if model is not None:
        del model
        model = None
        torch.cuda.empty_cache()


def _get_model():
    """获取模型实例，若未加载则按需加载（调用前必须先持有 model_lock）"""
    global last_request_time
    last_request_time = time.time()
    if model is None:
        _load_model()
    return model


def _idle_monitor():
    """后台线程：定期检查闲置时间，超时时自动卸载模型释放显存"""
    while True:
        time.sleep(60)
        with model_lock:
            if model is not None and time.time() - last_request_time > IDLE_TIMEOUT:
                _unload_model()


# 启动闲置监控线程
threading.Thread(target=_idle_monitor, daemon=True).start()
# ==============================================


@app.post("/transcribe")
async def transcribe(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(await file.read())
        path = f.name

    with model_lock:
        current_model = _get_model()
        segments, info = current_model.transcribe(path)
        text = ""
        for seg in segments:
            text += seg.text

    return {
        "language": info.language,
        "text": text
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=56178)
