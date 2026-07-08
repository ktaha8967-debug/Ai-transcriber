import os
import shutil
import uuid
import logging
import threading
from typing import List, Optional, Any
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel

import summarizer
import document_generator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Transcriber & Summarizer")

# Session structure
class SaveRequest(BaseModel):
    session_id: str
    text_segments: List[Any]
    language: Optional[str] = "en"

# ============================================================================
# WHISPER MODELS
# ============================================================================

whisper_models = {}
whisper_locks = {}
whisper_sizes = ["tiny", "base", "small"]
current_model_index = 0

def load_whisper_models():
    """Load all Whisper models in background."""
    global whisper_models, whisper_locks
    
    for size in whisper_sizes:
        try:
            logger.info(f"Loading Whisper {size}...")
            model = WhisperModel(size, device="cpu", compute_type="int8")
            whisper_models[size] = model
            whisper_locks[size] = threading.Lock()
            logger.info(f"Whisper {size} loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load Whisper {size}: {e}")

def get_next_model() -> str:
    """Get next model in round-robin."""
    global current_model_index
    available = [s for s in whisper_sizes if s in whisper_models]
    if not available:
        return None
    model_size = available[current_model_index % len(available)]
    current_model_index += 1
    return model_size

def transcribe_with_whisper(file_path: str, model_size: str, language: str = "auto"):
    """Transcribe audio using Whisper model."""
    model = whisper_models[model_size]
    
    transcribe_args = {
        "beam_size": 1,
        "vad_filter": True,
        "task": "translate"
    }
    
    if language and language != "auto":
        transcribe_args["language"] = language
    
    logger.info(f"Transcribing with Whisper {model_size}...")
    
    local_segments, info = model.transcribe(file_path, **transcribe_args)
    
    segments = []
    for seg in local_segments:
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        })
    
    return segments, info.language

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
    logger.info("Loading Whisper models...")
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
    return FileResponse(
        path=apk_path,
        filename="AuraScribe-Pro.apk",
        media_type="application/vnd.android.package-archive"
    )

@app.get("/api/pool-status")
async def get_pool_status():
    loaded = list(whisper_models.keys())
    return {"success": True, "status": {"models": loaded, "total": len(loaded)}}

@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("auto")
):
    """Transcribe audio using Whisper models."""
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"Received audio: {file_size} bytes")
        
        # Check if any model is loaded
        if not whisper_models:
            return JSONResponse(
                status_code=503,
                content={"success": False, "error": "Whisper models still loading. Please wait..."}
            )
        
        # Get next model
        model_size = get_next_model()
        if not model_size:
            return JSONResponse(
                status_code=503,
                content={"success": False, "error": "No Whisper model available."}
            )
        
        # Transcribe
        with whisper_locks[model_size]:
            segments, detected_language = transcribe_with_whisper(temp_file_path, model_size, language)
        
        logger.info(f"Success! Model: whisper-{model_size}, Segments: {len(segments)}")
        
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
            
            result_segments.append({
                "timestamp": timestamp,
                "text": text_val,
                "start": start_val,
                "end": end_val
            })
            full_text_list.append(text_val)
        
        return {
            "success": True,
            "detected_language": detected_language,
            "language_probability": 1.0,
            "segments": result_segments,
            "full_text": " ".join(full_text_list),
            "model_used": f"whisper-{model_size}"
        }
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
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
            summary, bullet_points = summarizer.summarize_text(full_text, num_sentences=4)
            summary_points = bullet_points
        else:
            summary_points = ["Audio transcription was too short to generate a summary."]
            
        docx_path = document_generator.save_to_docx(session_id, text_segments, summary_points)
        pdf_path = document_generator.save_to_pdf(session_id, summary_points, full_text)
        
        draft_path = os.path.join(TRANSCRIPTION_DIR, f"{session_id}_draft.txt")
        draft_lines = []
        for seg in text_segments:
            if isinstance(seg, dict):
                draft_lines.append(f"[{seg.get('timestamp', '')}] {seg.get('text', '')}")
            else:
                draft_lines.append(str(seg))
        
        with open(draft_path, "w", encoding="utf-8") as f:
            f.write("\n".join(draft_lines))
            
        return {
            "success": True,
            "session_id": session_id,
            "docx_path": docx_path,
            "pdf_path": pdf_path,
            "summary_points": summary_points
        }
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
                    except:
                        pass
                
                sessions.append({
                    "session_id": session_id,
                    "date": formatted_time,
                    "preview": preview,
                    "pdf_exists": pdf_exists,
                    "mtime": mtime
                })
                
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
