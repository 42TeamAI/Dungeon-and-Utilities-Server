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
import os
import httpx


def find_best_loop_point(audio, crossfade_samples):
    end_section = audio[-crossfade_samples:]
    start_section = audio[:crossfade_samples]

    correlation = correlate(start_section, end_section, mode='full')
    offset = np.argmax(correlation) - (crossfade_samples - 1)
    
    if offset < 0:
        offset = 0
    elif offset > len(audio) - crossfade_samples:
        offset = len(audio) - crossfade_samples
    
    return offset


def post_processing(audio, samplerate):

    crossfade_duration = 0.5  
    crossfade_samples = int(crossfade_duration * samplerate)

    if len(audio) < crossfade_samples:
        raise ValueError("Audio is too short for the specified crossfade duration.")


    if len(audio.shape) == 1:
        num_channels = 1
    else:
        num_channels = audio.shape[1]

 
    best_offset = find_best_loop_point(audio, crossfade_samples)

 
    start_fade = windows.tukey(crossfade_samples, alpha=0.5)
    end_fade = windows.tukey(crossfade_samples, alpha=0.5)[::-1]


    if num_channels > 1:
        start_fade = start_fade[:, np.newaxis]
        end_fade = end_fade[:, np.newaxis]

    start_section = audio[best_offset:best_offset + crossfade_samples]
    end_section = audio[-crossfade_samples:]

    faded_start = start_section * start_fade
    faded_end = end_section * end_fade
    blended_section = faded_start + faded_end


    blended_section /= np.max(np.abs(blended_section))

    looped_audio = np.concatenate((audio[:best_offset], audio[best_offset + crossfade_samples:] ))
    return looped_audio


class Data(BaseModel):
    description: str
    duration: int
    rephrase: bool

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json'), 'r') as f:
    config = json.load(f)
if config["proxy_url"]:
    chatgpt = OpenAI(api_key=config['ClosedAI'], http_client=httpx.Client(proxy=config["proxy_url"]))
else:
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
                print(config['ClosedAI'])
                return Response(status_code=501)
            try:
                prompt = chatgpt.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[

                        {"role": "user",
                        "content": f"Напиши описание музыки на английском языке для модели машинного обучения генерирующую музыку, которая создаст подходящую атмосферу для следующей ситуации в Dungeon and  Dragons: {data.description} Напиши только одно описание и ничего больше"}
                    ]
                ).choices[0].message.content
                print(prompt)
            except Exception as e:
                print(e)
                return Response(status_code=501)
        else:
            prompt = data.description
        print(data.description)
        res = model.generate(
            [prompt],
            progress=True
        ).cpu().numpy()[0, 0]
        res = post_processing(res)
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