import os
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="SignBridge optional model gateway", version="2.0.0")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
MODEL_API_URL = os.getenv("MODEL_API_URL", "").strip()
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "").strip()
MAX_CLIP_BYTES = int(os.getenv("MAX_CLIP_BYTES", str(25 * 1024 * 1024)))
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in FRONTEND_ORIGIN.split(",") if x.strip()], allow_credentials=False, allow_methods=["GET","POST","OPTIONS"], allow_headers=["*"])

@app.get("/api/health")
def health():
    return {"status":"ok","translation_model_configured":bool(MODEL_API_URL),"note":"The browser gesture model is separate from this optional sentence-translation gateway."}

@app.post("/api/translate")
async def translate(video: UploadFile = File(...), source_language: str = Form("ISL"), target_language: str = Form("English"), task: str = Form("sentence")):
    if not MODEL_API_URL:
        raise HTTPException(status_code=503, detail="No continuous-sign translation endpoint is configured. Set MODEL_API_URL in server/.env.")
    raw = await video.read()
    if not raw: raise HTTPException(status_code=400, detail="Empty video clip.")
    if len(raw) > MAX_CLIP_BYTES: raise HTTPException(status_code=413, detail=f"Clip exceeds {MAX_CLIP_BYTES} bytes.")
    headers = {"Authorization": f"Bearer {MODEL_API_KEY}"} if MODEL_API_KEY else {}
    files = {"video": (video.filename or "clip.webm", raw, video.content_type or "video/webm")}
    data = {"source_language":source_language,"target_language":target_language,"task":task}
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(MODEL_API_URL, headers=headers, files=files, data=data)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach model endpoint: {type(exc).__name__}")
    if response.status_code >= 400: raise HTTPException(status_code=502, detail=f"Model endpoint returned HTTP {response.status_code}.")
    try: payload = response.json()
    except ValueError: raise HTTPException(status_code=502, detail="Model endpoint must return JSON.")
    if not isinstance(payload,dict) or not payload.get("translation"):
        raise HTTPException(status_code=502, detail="Model endpoint must return a JSON object with a translation field.")
    return {"translation":payload.get("translation"),"confidence":payload.get("confidence"),"detected_language":payload.get("detected_language"),"language_confidence":payload.get("language_confidence"),"note":payload.get("note","Prediction is not guaranteed to be accurate.")}
