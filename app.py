
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"

app = FastAPI(title="SignBridge API")

origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGIN", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    language: str = Field(default="English", max_length=80)


class TranslationRequest(BaseModel):
    language: str = Field(default="auto", max_length=30)
    frames: list[list[float]] = Field(min_length=1, max_length=120)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "chat_configured": bool(os.getenv("OPENAI_API_KEY")),
        "translation_configured": bool(os.getenv("MODEL_API_URL")),
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Chatbot is not configured yet. Add OPENAI_API_KEY in Render."
        )

    try:
        client = AsyncOpenAI(api_key=api_key)
        response = await client.responses.create(
            model=model,
            instructions=(
                "You are SignBridge, a friendly and accessible assistant. "
                "Help users understand sign languages and communication access. "
                "Reply in the requested language when practical. "
                "Do not claim that text alone is a verified sign-language translation."
            ),
            input=(
                f"Preferred reply language: {request.language}\n"
                f"User message: {request.message}"
            ),
        )
        return {"reply": response.output_text}
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="The chatbot request failed. Check the Render logs and API configuration."
        )


@app.post("/api/translate")
async def translate(request: TranslationRequest):
    model_url = os.getenv("MODEL_API_URL")
    model_key = os.getenv("MODEL_API_KEY")

    if not model_url:
        raise HTTPException(
            status_code=503,
            detail=(
                "Sign recognition model is not connected yet. "
                "Configure MODEL_API_URL after deploying a compatible trained model."
            ),
        )

    headers = {}
    if model_key:
        headers["Authorization"] = f"Bearer {model_key}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            result = await client.post(
                model_url,
                json={
                    "language": request.language,
                    "frames": request.frames,
                },
                headers=headers,
            )
            result.raise_for_status()
            data: Any = result.json()
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="The sign recognition model could not process this sequence."
        )

    if not isinstance(data, dict) or not isinstance(
        data.get("translation"), str
    ):
        raise HTTPException(
            status_code=502,
            detail="The model must return JSON with a string 'translation' field."
        )

    return {
        "translation": data["translation"],
        "language": data.get("language", request.language),
    }


@app.get("/")
async def home():
    index_file = PUBLIC_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=404, detail="Website file not found")
    return FileResponse(index_file)


if PUBLIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")
