from fastapi import FastAPI
from pydantic import BaseModel
from threading import RLock
from audiocraft.models import musicgen
from openai import OpenAI
from fastapi.responses import Response
import io
import soundfile
import json
import uvicorn
import torch
from scipy.signal import correlate, windows
import numpy as np

class Data(BaseModel):
    description: str
    duration: int
    rephrase: bool

with open('config.json', 'r') as f:
    config = json.load(f)

chatgpt = OpenAI(api_key=config['ClosedAI'])
lock = RLock()
if config['GPU']:
    device = 'cuda'
else:
    device = 'cpu'
model = musicgen.MusicGen.get_pretrained(config['model'], device=device)
if device == 'cuda':
    model.compression_model.cpu()
    model.lm.cpu()
    torch.cuda.empty_cache()
app = FastAPI()

@app.get("/ping/")
async def ping():
    return Response(status_code=200)


@app.post("/music/", responses = {
        200: {
            "content": {f"audio/{config['format']}": {}}
        }
    },
    response_class=Response)
async def generate_music(data: Data):
    with lock:
        if device == 'cuda':
            model.compression_model.cuda()
            model.lm.cuda()
        model.set_generation_params(duration=data.duration)
        if data.rephrase:
            if config['ClosedAI'] == "API-key":
                return Response(status_code=501)
            prompt = chatgpt.chat.completions.create(
                model="gpt-4o-mini",
                messages=[

                    {"role": "user",
                     "content": f"Напиши короткое описание музыки на английском языке для модели машинного обучения генерирующую музыку, которая создаст подходящую атмосферу для следующей ситуации в Dungeon and  Dragons: {data.description} Напиши только одно описание и ничего больше"}
                ]
            ).choices[0].message

        else:
            prompt = data.description
        res = model.generate(
            [prompt],
            progress=True
        ).cpu().numpy()[0, 0]

        byte_io = io.BytesIO()
        soundfile.write(byte_io, data=res, samplerate=model.sample_rate, format= config['format'])
        byte_io.seek(0)
        if device == 'cuda':
            model.compression_model.cpu()
            model.lm.cpu()
            torch.cuda.empty_cache()
        return Response(content=byte_io.read(), media_type=f"audio/{config['format']}")


if __name__=='__main__':
        uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config['port'],
    )