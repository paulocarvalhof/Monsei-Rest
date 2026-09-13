import os
from threading import Thread
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Repositório correto identificado na sua imagem
model_id = "MidNurdos/Monsei-Atchk"
token = os.environ.get("HF_TOKEN")

print("Iniciando carregamento do modelo...")

# Carrega o tokenizador do Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

# Carrega o modelo com suporte a CPU e baixo consumo de RAM
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=token,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)


class MessageRequest(BaseModel):
    message: str


@app.post("/chat")
def chat_stream(request: MessageRequest):
    inputs = tokenizer(request.message, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=256, do_sample=True)
    thread = Thread(target=model.generate, kwargs=kwargs)
    thread.start()

    def generate_tokens():
        try:
            for token_text in streamer:
                yield token_text
        except Exception as e:
            yield f"\n[Erro: {str(e)}]"

    return StreamingResponse(generate_tokens(), media_type="text/plain")


@app.get("/")
def home():
    return {"status": "Monsei IA API Online", "model": model_id}
