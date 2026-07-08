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
import json as json_module

# Try importing vosk for fast local transcription
VOSK_AVAILABLE = False
try:
    from vosk import Model as VoskModel, KaldiRecognizer
    import wave
    VOSK_AVAILABLE = True
except ImportError:
    pass

import summarizer
import document_generator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Transcriber & Summarizer")

# Session structure to hold in-memory drafts
class SaveRequest(BaseModel):
    session_id: str
    text_segments: List[Any]
    language: Optional[str] = "en"

# ============================================================================
# MODEL POOL - Round-Robin Auto-Distribution System
# ============================================================================

class ModelPool:
    """
    Manages multiple transcription models and distributes work in round-robin fashion.
    """
    
    def __init__(self):
        self.models = {}
        self.model_locks = {}
        self.round_robin_index = 0
        self.lock = threading.Lock()
        self.model_order = []
        self.ready = False
        
    def register_vosk_model(self, name: str, model: Any):
        self.models[name] = model
        self.model_locks[name] = threading.Lock()
        self.model_order.append(name)
        logger.info(f"Registered Vosk model: {name}")
        
    def register_whisper_model(self, name: str, model: Any):
        self.models[name] = model
        self.model_locks[name] = threading.Lock()
        self.model_order.append(name)
        logger.info(f"Registered Whisper model: {name}")
        
    def get_next_model(self) -> str:
        with self.lock:
            if not self.model_order:
                return None
            model_name = self.model_order[self.round_robin_index % len(self.model_order)]
            self.round_robin_index += 1
            return model_name
            
    def transcribe(self, file_path: str, language: str = "auto") -> tuple:
        model_name = self.get_next_model()
        if not model_name:
            raise Exception("No models available in pool")
            
        logger.info(f"Assigning batch to model: {model_name}")
        
        with self.model_locks[model_name]:
            try:
                if model_name.startswith("vosk-"):
                    segments, lang = self._transcribe_vosk(file_path, model_name, language)
                elif model_name.startswith("whisper-"):
                    segments, lang = self._transcribe_whisper(file_path, model_name, language)
                else:
                    raise Exception(f"Unknown model type: {model_name}")
                    
                return segments, lang, model_name
            except Exception as e:
                logger.error(f"Model {model_name} failed: {e}")
                raise
                
    def _transcribe_vosk(self, file_path: str, model_name: str, language: str) -> tuple:
        model = self.models[model_name]
        
        wf = wave.open(file_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise Exception("Vosk requires mono 16-bit 16kHz WAV audio")
            
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json_module.loads(rec.Result())
                if result.get("result"):
                    for word_info in result["result"]:
                        results.append(word_info)
                        
        final = json_module.loads(rec.FinalResult())
        if final.get("result"):
            for word_info in final["result"]:
                results.append(word_info)
                
        wf.close()
        
        if not results:
            return [], language
            
        segments = []
        current_words = []
        current_start = results[0]["start"]
        
        for word_info in results:
            current_words.append(word_info)
            if (word_info["end"] - current_start >= 3.0) or \
               (len(current_words) > 0 and word_info["end"] - current_words[-1]["start"] > 1.5):
                text = " ".join([w["word"] for w in current_words])
                segments.append({
                    "start": current_start,
                    "end": word_info["end"],
                    "text": text.strip()
                })
                current_words = []
                current_start = word_info["end"]
                
        if current_words:
            text = " ".join([w["word"] for w in current_words])
            segments.append({
                "start": current_start,
                "end": current_words[-1]["end"],
                "text": text.strip()
            })
            
        lang = model_name.replace("vosk-small-", "")
        return segments, lang
        
    def _transcribe_whisper(self, file_path: str, model_name: str, language: str) -> tuple:
        model = self.models[model_name]
        
        transcribe_args = {
            "beam_size": 1,
            "vad_filter": True,
            "task": "translate"
        }
        if language != "auto":
            transcribe_args["language"] = language
            
        local_segments, info = model.transcribe(file_path, **transcribe_args)
        
        segments = []
        for seg in local_segments:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text
            })
            
        return segments, info.language
        
    def get_status(self) -> dict:
        return {
            "total_models": len(self.model_order),
            "models": self.model_order,
            "next_index": self.round_robin_index % max(len(self.model_order), 1)
        }

# Global model pool
model_pool = ModelPool()

# ============================================================================
# VOSK MODEL DOWNLOAD AND REGISTRATION
# ============================================================================

VOSK_MODEL_URLS = {
    "vosk-small-en": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "vosk-small-hi": "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip",
    "vosk-small-es": "https://alphacephei.com/vosk/models/vosk-model-small-es-0.22.zip",
    "vosk-small-fr": "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
    "vosk-small-de": "https://alphacephei.com/vosk/models/vosk-model-small-de-0.22.zip",
    "vosk-small-cn": "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip",
    "vosk-small-ar": "https://alphacephei.com/vosk/models/vosk-model-arabic-0.22-linto12-5.0.zip",
    "vosk-small-pt": "https://alphacephei.com/vosk/models/vosk-model-small-pt-0.22.zip",
}

def download_vosk_model(lang_key: str) -> str:
    import zipfile
    import urllib.request
    
    model_dir = os.path.join("vosk_models", lang_key)
    if os.path.exists(model_dir) and os.listdir(model_dir):
        return model_dir
        
    url = VOSK_MODEL_URLS.get(lang_key)
    if not url:
        raise Exception(f"No URL for Vosk model: {lang_key}")
        
    os.makedirs("vosk_models", exist_ok=True)
    zip_path = os.path.join("vosk_models", f"{lang_key}.zip")
    
    logger.info(f"Downloading Vosk model {lang_key}...")
    urllib.request.urlretrieve(url, zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("vosk_models")
    os.remove(zip_path)
    
    logger.info(f"Vosk model {lang_key} downloaded.")
    return model_dir

def init_models_background():
    import concurrent.futures
    
    def load_vosk_model(lang_key):
        try:
            model_path = download_vosk_model(lang_key)
            model = VoskModel(model_path)
            model_pool.register_vosk_model(lang_key, model)
            logger.info(f"Vosk {lang_key} ready!")
        except Exception as e:
            logger.error(f"Failed to load Vosk {lang_key}: {e}")
            
    def load_whisper_model(size):
        try:
            model = WhisperModel(size, device="cpu", compute_type="int8")
            model_pool.register_whisper_model(f"whisper-{size}", model)
            logger.info(f"Whisper {size} ready!")
        except Exception as e:
            logger.error(f"Failed to load Whisper {size}: {e}")
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = []
        
        for lang_key in VOSK_MODEL_URLS.keys():
            futures.append(executor.submit(load_vosk_model, lang_key))
            
        for size in ["tiny", "base", "small"]:
            futures.append(executor.submit(load_whisper_model, size))
            
        concurrent.futures.wait(futures)
        
    model_pool.ready = True
    logger.info(f"All models loaded! Pool has {len(model_pool.model_order)} models.")

# ============================================================================
# TRANSCRIPTION ENDPOINTS
# ============================================================================

TRANSCRIPTION_DIR = "transcriptions"
if not os.path.exists(TRANSCRIPTION_DIR):
    os.makedirs(TRANSCRIPTION_DIR)

STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting model pool initialization...")
    threading.Thread(target=init_models_background, daemon=True).start()

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
    return {"success": True, "status": model_pool.get_status()}

@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("auto")
):
    """
    Transcribes audio using automatic model distribution.
    Uses Vosk (fast) and Whisper (accurate) models.
    """
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
    
    segments = None
    detected_language = "auto"
    language_probability = 1.0
    model_used = "unknown"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"Received audio: {file_size} bytes")
        
        # Use Model Pool (Vosk + Whisper auto-distribution)
        if model_pool.ready:
            try:
                logger.info("Using Model Pool...")
                segments, detected_language, model_used = model_pool.transcribe(temp_file_path, language)
                logger.info(f"Success! Model: {model_used}, Segments: {len(segments)}")
            except Exception as e:
                logger.warning(f"Model Pool failed: {e}")
        
        # Fallback: Basic Whisper
        if segments is None:
            logger.info("Fallback: Loading Whisper tiny...")
            try:
                model = WhisperModel("tiny", device="cpu", compute_type="int8")
                transcribe_args = {"beam_size": 1, "vad_filter": True, "task": "translate"}
                if language != "auto":
                    transcribe_args["language"] = language
                    
                local_segments, info = model.transcribe(temp_file_path, **transcribe_args)
                segments = [{"start": s.start, "end": s.end, "text": s.text} for s in local_segments]
                detected_language = info.language
                language_probability = info.language_probability
                model_used = "whisper-tiny-fallback"
            except Exception as e:
                logger.error(f"All methods failed: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "error": "All transcription models failed."}
                )
        
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
            
        logger.info(f"Transcription complete. Model: {model_used}, Language: {detected_language}")
        
        return {
            "success": True,
            "detected_language": detected_language,
            "language_probability": language_probability,
            "segments": result_segments,
            "full_text": " ".join(full_text_list),
            "model_used": model_used
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
            
        logger.info(f"Saving session: {session_id}...")
        
        docx_path = document_generator.save_to_docx(session_id, text_segments, summary_points)
        pdf_path = document_generator.save_to_pdf(session_id, summary_points, full_text)
        
        draft_path = os.path.join(TRANSCRIPTION_DIR, f"{session_id}_draft.txt")
        draft_lines = []
        for seg in text_segments:
            if isinstance(seg, dict):
                draft_lines.append(f"[{seg.get('timestamp', '')}] {seg.get('text', '')}")
            else:
                draft_lines.append(str(seg))
        draft_content = "\n".join(draft_lines)
        
        with open(draft_path, "w", encoding="utf-8") as f:
            f.write(draft_content)
            
        return {
            "success": True,
            "session_id": session_id,
            "docx_path": docx_path,
            "pdf_path": pdf_path,
            "summary_points": summary_points
        }
    except Exception as e:
        logger.error(f"Save error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/sessions")
async def get_sessions():
    try:
        files = os.listdir(TRANSCRIPTION_DIR)
        sessions = []
        for file in files:
            if file.endswith(".docx"):
                session_id = file[:-5]
                docx_path = os.path.join(TRANSCRIPTION_DIR, file)
                pdf_file = f"{session_id}_summary.pdf"
                pdf_exists = pdf_file in files
                
                mtime = os.path.getmtime(docx_path)
                formatted_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                preview = ""
                draft_file = f"{session_id}_draft.txt"
                if draft_file in files:
                    try:
                        with open(os.path.join(TRANSCRIPTION_DIR, draft_file), "r", encoding="utf-8") as f:
                            preview = f.read(150) + "..."
                    except Exception:
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
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

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
