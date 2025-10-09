from fastapi import FastAPI
from pydantic import BaseModel
from inference import *
import uvicorn
from typing import Optional, List

app = FastAPI(title="MT EN→VI", version="1.0")
translator = Translator(model_dir="../checkpoints/checkpoint-49995")  # adjust path in Docker/production

class TranslateRequest(BaseModel):
    texts: List[str]
    num_beams: Optional[int] = 5
    max_length: Optional[int] = 256

class TranslateResponse(BaseModel):
    translations: List[str]

@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    outs = []
    for t in req.texts:
        outs.append(translator.translate(t, max_length=req.max_length, num_beams=req.num_beams))
    return TranslateResponse(translations=outs)

@app.get("/health")
def health():
    return {"status":"ok"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
