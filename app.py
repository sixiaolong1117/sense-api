from fastapi import FastAPI, UploadFile
from faster_whisper import WhisperModel
import tempfile

app = FastAPI()

model = WhisperModel(
    "large-v3-turbo",
    device="cuda",
    compute_type="float16"
)

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(await file.read())
        path = f.name

    segments, info = model.transcribe(path)
    text = ""
    for seg in segments:
        text += seg.text

    return {
        "language": info.language,
        "text": text
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
