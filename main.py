import os
import logging
import traceback

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import InferenceClient
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("monsei")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_id = "MidNurdos/Monsei-Atchk"
token = os.environ.get("HF_TOKEN")

client = InferenceClient(
    model=model_id,
    token=token
)

class MessageRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_stream(request: MessageRequest):

    def generate_tokens():
        try:
            for response in client.text_generation(
                request.message,
                max_new_tokens=256,
                stream=True
            ):
                yield response

        except Exception as e:
            logger.error("ERRO NO MODELO:")
            logger.error("Tipo: %s", type(e).__name__)
            logger.error("Mensagem: %r", str(e))
            logger.error(traceback.format_exc())

            yield "\n[Erro interno do modelo. Veja os logs do servidor.]"

    return StreamingResponse(
        generate_tokens(),
        media_type="text/plain"
    )

@app.get("/")
def home():
    return {"status": "Monsei IA Online"}
