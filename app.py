import logging
from fastapi import FastAPI, UploadFile
from faster_whisper import WhisperModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import io
import tempfile
import time
import threading
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sense-api")

app = FastAPI()

# =====================================================================
#  通用模型管理器 - 支持闲置超时自动卸载显存
# =====================================================================
class ModelManager:
    """通用模型管理器

    每个模型独立管理，各自拥有：
    - 独立的线程锁
    - 独立的闲置超时阈值
    - 独立的后台监控线程
    """

    def __init__(self, name, load_fn, unload_fn, idle_timeout=300):
        self.name = name
        self._load_fn = load_fn
        self._unload_fn = unload_fn
        self.idle_timeout = idle_timeout
        self.lock = threading.Lock()
        self._model = None
        self._last_access = time.time()
        threading.Thread(
            target=self._monitor_loop, daemon=True, name=f"{name}-monitor"
        ).start()

    def get_model(self):
        """获取模型实例，若未加载则按需加载（调用方需先获取 self.lock）"""
        self._last_access = time.time()
        if self._model is None:
            log.info("%s: 开始加载模型（若未缓存将下载，请耐心等待）...", self.name)
            self._model = self._load_fn()
            log.info("%s: 模型加载完成", self.name)
        return self._model

    def unload(self):
        """卸载模型并释放 GPU 显存（调用方需先获取 self.lock）"""
        if self._model is not None:
            log.info("%s: 闲置超时，卸载模型释放显存", self.name)
            self._unload_fn(self._model)
            self._model = None
            torch.cuda.empty_cache()

    def _monitor_loop(self):
        while True:
            time.sleep(60)
            with self.lock:
                if (
                    self._model is not None
                    and time.time() - self._last_access > self.idle_timeout
                ):
                    self.unload()


# ---------- Whisper 语音识别 ----------
def _load_whisper():
    return WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")


def _unload_whisper(model):
    del model


whisper_manager = ModelManager(
    name="whisper",
    load_fn=_load_whisper,
    unload_fn=_unload_whisper,
    idle_timeout=300,
)


# ---------- Qwen2-VL 视觉识别 ----------
QWEN_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"


def _load_qwen():
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return {"model": model, "processor": processor}


def _unload_qwen(model_dict):
    del model_dict["model"]
    del model_dict["processor"]


qwen_manager = ModelManager(
    name="qwen2-vl",
    load_fn=_load_qwen,
    unload_fn=_unload_qwen,
    idle_timeout=300,
)
# =====================================================================


@app.post("/transcribe")
async def transcribe(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(await file.read())
        path = f.name

    with whisper_manager.lock:
        model = whisper_manager.get_model()
        segments, info = model.transcribe(path)
        text = ""
        for seg in segments:
            text += seg.text

    return {
        "language": info.language,
        "text": text,
    }


@app.post("/vision")
async def vision(file: UploadFile, prompt: str = "请描述这张图片的内容"):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    with qwen_manager.lock:
        model_dict = qwen_manager.get_model()
        model = model_dict["model"]
        processor = model_dict["processor"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_input], images=[image], padding=True, return_tensors="pt"
        ).to(model.device)

        output_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids = [
            output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))
        ]
        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return {"text": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=56178)
