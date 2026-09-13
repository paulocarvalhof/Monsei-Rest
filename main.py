import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import InferenceClient
from pydantic import BaseModel

app = FastAPI()

# Permite acesso do seu app (Flutter, Web, Mobile, Desktop)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega token e ID do seu modelo privado
token = os.environ.get("HF_TOKEN")
model_id = "seu-usuario/nome-do-modelo-privado"  # Subtitua pelo seu repositório no HF

# Cliente oficial de inferência em nuvem do Hugging Face
client = InferenceClient(model=model_id, token=token)


class MessageRequest(BaseModel):
    message: str


@app.post("/chat")
def chat_stream(request: MessageRequest):
    def generate_tokens():
        try:
            # Faz streaming de texto direto do Hugging Face
            for chunk in client.text_generation(
                request.message, stream=True, max_new_tokens=256
            ):
                yield chunk
        except Exception as e:
            yield f"\n[Erro na geração: {str(e)}]"

    return StreamingResponse(generate_tokens(), media_type="text/plain")


@app.get("/")
def home():
    return {"status": "API online e conectada ao Render!"}

