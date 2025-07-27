# main.py
from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from medgpt import medgpt, reset_history, initialize_medgpt_model, is_initialized


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/ping")
async def ping():
    initialize_medgpt_model()
    return {"status": "ready"}


@app.get("/")
def root():
    return {"message": "Hello World"}



# Request & Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str


class RefreshFlag(BaseModel):
    flag: str


@app.post("/api/refresh")
async def receive_refresh(flag: RefreshFlag):
    # Perform any action you like (e.g., reset session, log, etc.)
    # print(f"Received refresh flag: {flag.flag}")
    reset_history()
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        answer = medgpt(req.message)

        return ChatResponse(reply=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
