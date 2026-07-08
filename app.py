import os
import shutil
import uuid
import logging
import threading
import base64
import json
from typing import List, Optional, Any
from datetime import datetime
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel

import document_generator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Transcriber & Summarizer")

# API Keys - Load from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Session structure
class SaveRequest(BaseModel):
    session_id: str
    text_segments: List[Any]
    language: Optional[str] = "en"

# ============================================================================
# GEMINI 2.5 TRANSCRIPTION (PRIMARY)
# ============================================================================

def transcribe_via_gemini(file_path: str, language: str = "auto"):
    """
    Transcribes audio using Gemini 2.5 Flash (FREE).
    This is the PRIMARY model - tried first.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    
    with open(file_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode("utf-8")
    
    lang_instruction = ""
    if language and language != "auto":
        lang_map = {"en": "English", "hi": "Hindi", "ur": "Urdu", "bn": "Bengali", 
                    "es": "Spanish", "fr": "French", "de": "German", "zh": "Chinese", 
                    "ar": "Arabic", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese", "ko": "Korean"}
        lang_instruction = f"Transcribe in {lang_map.get(language, language)}."
    
    prompt = f"""Transcribe this audio accurately. {lang_instruction}

Return ONLY a JSON array with segments. Each segment must have:
- "start": float (start time in seconds)
- "end": float (end time in seconds)
- "text": string (transcribed text)

Example: [{{"start": 0.0, "end": 2.5, "text": "Hello world"}}]

Return ONLY the JSON array, nothing else."""
    
    payload = {
        "contents": [{
            "parts": [
                {"inlineData": {"mimeType": "audio/wav", "data": audio_data}},
                {"text": prompt}
            ]
        }],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)
            
            if response.status_code != 200:
                raise Exception(f"Gemini error {response.status_code}: {response.text[:200]}")
            
            resp_data = response.json()
            text_content = resp_data["candidates"][0]["content"]["parts"][0]["text"]
            
            segments = json.loads(text_content)
            if not isinstance(segments, list):
                segments = []
            
            return segments, language if language != "auto" else "en"
            
    except Exception as e:
        logger.warning(f"Gemini failed: {e}")
        raise

# ============================================================================
# WHISPER MODELS (FALLBACK)
# ============================================================================

whisper_models = {}
whisper_locks = {}
whisper_sizes = ["tiny", "base", "small"]
current_model_index = 0

def load_whisper_models():
    global whisper_models, whisper_locks
    for size in whisper_sizes:
        try:
            logger.info(f"Loading Whisper {size}...")
            model = WhisperModel(size, device="cpu", compute_type="int8")
            whisper_models[size] = model
            whisper_locks[size] = threading.Lock()
            logger.info(f"Whisper {size} ready!")
        except Exception as e:
            logger.error(f"Failed to load Whisper {size}: {e}")

def get_next_whisper() -> str:
    global current_model_index
    available = [s for s in whisper_sizes if s in whisper_models]
    if not available:
        return None
    model_size = available[current_model_index % len(available)]
    current_model_index += 1
    return model_size

def transcribe_with_whisper(file_path: str, model_size: str, language: str = "auto"):
    model = whisper_models[model_size]
    transcribe_args = {"beam_size": 1, "vad_filter": True, "task": "translate"}
    if language and language != "auto":
        transcribe_args["language"] = language
    
    local_segments, info = model.transcribe(file_path, **transcribe_args)
    segments = [{"start": s.start, "end": s.end, "text": s.text} for s in local_segments]
    return segments, info.language

# ============================================================================
# GROQ SUMMARIZATION
# ============================================================================

def summarize_with_groq(text: str, num_sentences: int = 4):
    """Generate summary using Groq API."""
    if not GROQ_API_KEY:
        return None, None
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    prompt = f"""Summarize the following transcription into {num_sentences} key points.
Return as a JSON array of strings, each being one key point.

Text: {text[:3000]}

Return ONLY a JSON array like: ["Point 1", "Point 2", "Point 3", "Point 4"]"""
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                # Extract JSON from response
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    points = json.loads(json_match.group())
                    return points, points
    except Exception as e:
        logger.warning(f"Groq summary failed: {e}")
    
    return None, None

def summarize_text(text, num_sentences=4):
    """Summarize text - tries Groq first, falls back to local."""
    # Try Groq API first
    summary, points = summarize_with_groq(text, num_sentences)
    if summary:
        logger.info("Summary generated via Groq API")
        return summary, points
    
    # Fallback to local summarizer
    try:
        import re
        import numpy as np
        
        def split_into_sentences(text):
            sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|۔|\|)\s*')
            sentences = sentence_endings.split(text)
            return [s.strip() for s in sentences if len(s.strip()) > 10]
        
        sentences = split_into_sentences(text)
        if len(sentences) <= num_sentences:
            return sentences, sentences
        
        # Simple extractive summarization
        ranked = sorted(range(len(sentences)), key=lambda i: len(sentences[i]), reverse=True)
        top_indices = sorted(ranked[:num_sentences])
        summary = [sentences[i] for i in top_indices]
        bullet_points = [sentences[i] for i in ranked[:num_sentences + 2]]
        
        logger.info("Summary generated locally")
        return summary, bullet_points
    except Exception as e:
        logger.error(f"Local summary failed: {e}")
        return ["Summary generation failed"], ["Summary generation failed"]

# ============================================================================
# API ENDPOINTS
# ============================================================================

TRANSCRIPTION_DIR = "transcriptions"
if not os.path.exists(TRANSCRIPTION_DIR):
    os.makedirs(TRANSCRIPTION_DIR)

STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

@app.on_event("startup")
async def startup_event():
    logger.info("Loading Whisper models (fallback)...")
    threading.Thread(target=load_whisper_models, daemon=True).start()

@app.get("/")
async def root():
    return RedirectResponse(url="/landing.html")

@app.get("/app")
async def app_redirect():
    return RedirectResponse(url="/index.html")

@app.get("/download-apk")
async def download_apk():
    apk_path = os.path.join(os.path.dirname(__file__), "app-debug.apk")
    if not os.path.exists(apk_path):
        raise HTTPException(status_code=404, detail="APK file not found")
    return FileResponse(path=apk_path, filename="AuraScribe-Pro.apk", media_type="application/vnd.android.package-archive")

@app.get("/api/pool-status")
async def get_pool_status():
    whisper_loaded = list(whisper_models.keys())
    return {"success": True, "status": {"whisper": whisper_loaded, "gemini": "ready"}}

@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("auto")
):
    """Transcribe audio - Gemini 2.5 first, Whisper fallback."""
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"Received audio: {file_size} bytes")
        
        segments = None
        detected_language = "auto"
        model_used = "unknown"
        
        # PRIORITY 1: Gemini 2.5 Flash (FREE & FAST)
        try:
            logger.info("Trying Gemini 2.5 Flash...")
            segments, detected_language = transcribe_via_gemini(temp_file_path, language)
            model_used = "gemini-2.5-flash"
            logger.info(f"Gemini SUCCESS! Segments: {len(segments)}")
        except Exception as e:
            logger.warning(f"Gemini failed: {e}")
        
        # PRIORITY 2: Whisper (if Gemini failed)
        if segments is None and whisper_models:
            model_size = get_next_whisper()
            if model_size:
                try:
                    logger.info(f"Trying Whisper {model_size}...")
                    with whisper_locks[model_size]:
                        segments, detected_language = transcribe_with_whisper(temp_file_path, model_size, language)
                    model_used = f"whisper-{model_size}"
                    logger.info(f"Whisper SUCCESS! Segments: {len(segments)}")
                except Exception as e:
                    logger.warning(f"Whisper failed: {e}")
        
        # No model worked
        if segments is None:
            return JSONResponse(status_code=500, content={"success": False, "error": "All transcription models failed."})
        
        # Format segments
        result_segments = []
        full_text_list = []
        for segment in segments:
            start_val = segment.get("start", 0.0)
            end_val = segment.get("end", 0.0)
            text_val = segment.get("text", "").strip()
            
            start_min = int(start_val // 60)
            start_sec = int(start_val % 60)
            end_min = int(end_val // 60)
            end_sec = int(end_val % 60)
            timestamp = f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
            
            result_segments.append({"timestamp": timestamp, "text": text_val, "start": start_val, "end": end_val})
            full_text_list.append(text_val)
        
        return {
            "success": True,
            "detected_language": detected_language,
            "language_probability": 1.0,
            "segments": result_segments,
            "full_text": " ".join(full_text_list),
            "model_used": model_used
        }
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/save")
async def save_transcription(request: SaveRequest):
    try:
        session_id = request.session_id
        text_segments = request.text_segments
        
        full_text = " ".join([seg.get("text", "") if isinstance(seg, dict) else str(seg) for seg in text_segments])
        
        summary_points = []
        if len(full_text.strip()) > 30:
            summary, bullet_points = summarize_text(full_text, num_sentences=4)
            summary_points = bullet_points
        else:
            summary_points = ["Audio transcription was too short to generate a summary."]
            
        docx_path = document_generator.save_to_docx(session_id, text_segments, summary_points)
        pdf_path = document_generator.save_to_pdf(session_id, summary_points, full_text)
        
        draft_path = os.path.join(TRANSCRIPTION_DIR, f"{session_id}_draft.txt")
        draft_lines = [f"[{seg.get('timestamp', '')}] {seg.get('text', '')}" for seg in text_segments if isinstance(seg, dict)]
        
        with open(draft_path, "w", encoding="utf-8") as f:
            f.write("\n".join(draft_lines))
            
        return {"success": True, "session_id": session_id, "docx_path": docx_path, "pdf_path": pdf_path, "summary_points": summary_points}
    except Exception as e:
        logger.error(f"Save error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/api/sessions")
async def get_sessions():
    try:
        files = os.listdir(TRANSCRIPTION_DIR)
        sessions = []
        for file in files:
            if file.endswith(".docx"):
                session_id = file[:-5]
                docx_path = os.path.join(TRANSCRIPTION_DIR, file)
                pdf_exists = f"{session_id}_summary.pdf" in files
                mtime = os.path.getmtime(docx_path)
                formatted_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                preview = ""
                draft_file = f"{session_id}_draft.txt"
                if draft_file in files:
                    try:
                        with open(os.path.join(TRANSCRIPTION_DIR, draft_file), "r", encoding="utf-8") as f:
                            preview = f.read(150) + "..."
                    except: pass
                
                sessions.append({"session_id": session_id, "date": formatted_time, "preview": preview, "pdf_exists": pdf_exists, "mtime": mtime})
                
        sessions.sort(key=lambda x: x["mtime"], reverse=True)
        return {"success": True, "sessions": sessions}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/api/download/{file_type}/{session_id}")
async def download_file(file_type: str, session_id: str):
    if file_type == "docx":
        filename = f"{session_id}.docx"
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif file_type == "pdf":
        filename = f"{session_id}_summary.pdf"
        media_type = "application/pdf"
    else:
        raise HTTPException(status_code=400, detail="Invalid file type.")
        
    file_path = os.path.join(TRANSCRIPTION_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {filename} not found.")
        
    return FileResponse(path=file_path, filename=filename, media_type=media_type)

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
