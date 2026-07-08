import os
import shutil
import uuid
import logging
import threading
from typing import List, Optional, Any
from datetime import datetime
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel
import json as json_module
import queue

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

# Load .env file manually
def load_env():
    if os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ[key.strip()] = val.strip().strip('"').strip("'")
        except Exception as e:
            logger.warning(f"Failed to load .env file: {e}")

load_env()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
KIMI_AVAILABLE = True

# Nova Dynamics API Configuration
NOVA_API_KEY = "sk-nova-998877"
NOVA_API_URL = "https://britsyncuk--ollama-gpu-bench-run.modal.run/api/generate"

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
    When one model is busy transcribing a 3-second batch, the next batch goes to the next available model.
    """
    
    def __init__(self):
        self.models = {}  # model_name -> model_instance
        self.model_locks = {}  # model_name -> threading.Lock
        self.round_robin_index = 0
        self.lock = threading.Lock()
        self.model_order = []  # Ordered list of available model names
        self.ready = False
        
    def register_vosk_model(self, name: str, model: Any):
        """Register a Vosk model in the pool."""
        self.models[name] = model
        self.model_locks[name] = threading.Lock()
        self.model_order.append(name)
        logger.info(f"Registered Vosk model: {name}")
        
    def register_whisper_model(self, name: str, model: Any):
        """Register a Whisper model in the pool."""
        self.models[name] = model
        self.model_locks[name] = threading.Lock()
        self.model_order.append(name)
        logger.info(f"Registered Whisper model: {name}")
        
    def get_next_model(self) -> str:
        """Get the next model name in round-robin fashion."""
        with self.lock:
            if not self.model_order:
                return None
            model_name = self.model_order[self.round_robin_index % len(self.model_order)]
            self.round_robin_index += 1
            return model_name
            
    def transcribe(self, file_path: str, language: str = "auto") -> tuple:
        """
        Transcribe audio using the next available model in round-robin.
        Returns (segments, detected_language, model_used)
        """
        model_name = self.get_next_model()
        if not model_name:
            raise Exception("No models available in pool")
            
        logger.info(f"Assigning batch to model: {model_name}")
        
        # Acquire lock for this specific model (thread-safe)
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
                logger.error(f"Model {model_name} transcription failed: {e}")
                raise
                
    def _transcribe_vosk(self, file_path: str, model_name: str, language: str) -> tuple:
        """Transcribe using a Vosk model."""
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
            
        # Group words into segments
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
            
        # Extract language from model name
        lang = model_name.replace("vosk-small-", "")
        return segments, lang
        
    def _transcribe_whisper(self, file_path: str, model_name: str, language: str) -> tuple:
        """Transcribe using a Whisper model."""
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
        """Get pool status for debugging."""
        return {
            "total_models": len(self.model_order),
            "models": self.model_order,
            "next_index": self.round_robin_index % max(len(self.model_order), 1)
        }

# Global model pool
model_pool = ModelPool()

# ============================================================================
# MODEL DOWNLOAD AND REGISTRATION
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
    """Downloads and extracts a Vosk model if not already present."""
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
    
    logger.info(f"Vosk model {lang_key} downloaded successfully.")
    return model_dir

def init_models_background():
    """Initialize all models in background threads for faster startup."""
    import concurrent.futures
    
    def load_vosk_model(lang_key):
        """Load a single Vosk model."""
        try:
            model_path = download_vosk_model(lang_key)
            model = VoskModel(model_path)
            model_pool.register_vosk_model(lang_key, model)
            logger.info(f"Vosk model {lang_key} ready!")
        except Exception as e:
            logger.error(f"Failed to load Vosk model {lang_key}: {e}")
            
    def load_whisper_model(size):
        """Load a single Whisper model."""
        try:
            model = WhisperModel(size, device="cpu", compute_type="int8")
            model_pool.register_whisper_model(f"whisper-{size}", model)
            logger.info(f"Whisper model {size} ready!")
        except Exception as e:
            logger.error(f"Failed to load Whisper model {size}: {e}")
            
    # Load all models in parallel using thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = []
        
        # Submit Vosk models (8 models)
        for lang_key in VOSK_MODEL_URLS.keys():
            futures.append(executor.submit(load_vosk_model, lang_key))
            
        # Submit Whisper models (3 fast models)
        for size in ["tiny", "base", "small"]:
            futures.append(executor.submit(load_whisper_model, size))
            
        # Wait for all to complete
        concurrent.futures.wait(futures)
        
    model_pool.ready = True
    logger.info(f"All models loaded! Pool has {len(model_pool.model_order)} models ready.")

# ============================================================================
# CLOUD API TRANSCRIPTION FUNCTIONS
# ============================================================================

def transcribe_via_nova(file_path: str, language: str = "auto"):
    """
    Transcribes audio using Nova Dynamics GPU API (qwen2.5:32b).
    This is the PRIMARY model - tried first before all others.
    """
    import base64
    
    # Read audio file and encode to base64
    with open(file_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode("utf-8")
    
    # Prepare prompt for transcription
    lang_instruction = ""
    if language != "auto":
        lang_map = {"en": "English", "hi": "Hindi", "ur": "Urdu", "es": "Spanish", 
                    "fr": "French", "de": "German", "zh": "Chinese", "ar": "Arabic"}
        lang_instruction = f"Transcribe in {lang_map.get(language, language)}."
    
    prompt = f"""You are a professional audio transcription AI. Transcribe the following audio accurately.

{lang_instruction}

Return the transcription as a JSON array with segments. Each segment must have:
- "start": float (start time in seconds)
- "end": float (end time in seconds)  
- "text": string (transcribed text)

Return ONLY the JSON array, no other text.

Audio data (base64): {audio_data[:1000]}..."""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {NOVA_API_KEY}"
    }
    
    payload = {
        "model": "qwen2.5:32b",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(NOVA_API_URL, headers=headers, json=payload)
            if response.status_code != 200:
                raise Exception(f"Nova Dynamics API error: {response.text}")
            
            resp_data = response.json()
            ai_response = resp_data.get("response", "")
            
            # Try to parse JSON from response
            import re
            json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
            if json_match:
                segments = json_module.loads(json_match.group())
                if isinstance(segments, list) and len(segments) > 0:
                    return segments, language if language != "auto" else "en"
            
            # If no JSON found, return the text as a single segment
            return [{"start": 0.0, "end": 3.0, "text": ai_response.strip()}], "en"
            
    except Exception as e:
        logger.warning(f"Nova Dynamics API failed: {e}")
        raise

def transcribe_via_groq(file_path: str, api_key: str, language: str = "auto"):
    """Translates audio file using Groq Whisper API."""
    url = "https://api.groq.com/openai/v1/audio/translations"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    files = {"file": (os.path.basename(file_path), open(file_path, "rb"), "audio/wav")}
    data = {"model": "whisper-large-v3-turbo", "response_format": "verbose_json"}
    
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, headers=headers, files=files, data=data)
            if response.status_code != 200:
                raise Exception(f"Groq API error: {response.text}")
            
            resp_data = response.json()
            segments = resp_data.get("segments", [])
            detected_language = resp_data.get("language", "auto")
            
            parsed_segments = []
            for seg in segments:
                parsed_segments.append({
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "text": seg.get("text", "")
                })
            
            lang_mapping = {
                "urdu": "ur", "hindi": "hi", "english": "en", "spanish": "es",
                "french": "fr", "german": "de", "chinese": "zh", "arabic": "ar"
            }
            detected_language_code = lang_mapping.get(detected_language.lower(), detected_language)
            
            return parsed_segments, detected_language_code
    finally:
        if "file" in files:
            files["file"][1].close()

def transcribe_via_gemini(file_path: str, api_key: str):
    """Translates audio file using Google Gemini API."""
    import base64
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    with open(file_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode("utf-8")
        
    payload = {
        "contents": [{
            "parts": [
                {"inlineData": {"mimeType": "audio/wav", "data": audio_data}},
                {"text": (
                    "Translate the spoken audio content into English. Return the result strictly as a JSON array of segment objects. "
                    "Each object must contain three keys: 'start' (float, start time in seconds), "
                    "'end' (float, end time in seconds), and 'text' (string, translated English text). "
                    "Do not include any markdown formatting (like ```json), return only raw JSON."
                )}
            ]
        }],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    
    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, json=payload)
        if response.status_code != 200:
            raise Exception(f"Gemini API error: {response.text}")
            
        resp_data = response.json()
        try:
            text_content = resp_data["candidates"][0]["content"]["parts"][0]["text"]
            import json
            segments = json.loads(text_content)
            if not isinstance(segments, list):
                segments = []
            return segments, "auto"
        except Exception as e:
            raise Exception(f"Failed to parse Gemini response: {e}. Raw: {response.text}")

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
    """Initialize all models in background on startup."""
    logger.info("Starting model pool initialization in background...")
    threading.Thread(target=init_models_background, daemon=True).start()

@app.get("/api/pool-status")
async def get_pool_status():
    """Get current model pool status."""
    return {"success": True, "status": model_pool.get_status()}

@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("auto")
):
    """
    Transcribes the uploaded audio file using automatic model distribution.
    Backend picks the next available model automatically - no user selection needed.
    """
    load_env()
    groq_api_key = os.environ.get("GROQ_API_KEY")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
    
    segments = None
    detected_language = "auto"
    language_probability = 1.0
    model_used = "unknown"
    
    global KIMI_AVAILABLE
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Saved temporary file to {temp_file_path}")
        
        # PRIORITY 1: Nova Dynamics GPU API (PRIMARY MODEL - tried first!)
        try:
            logger.info("Trying Nova Dynamics GPU API (PRIMARY)...")
            segments, detected_language = transcribe_via_nova(temp_file_path, language)
            model_used = "nova-dynamics-gpu"
            logger.info("Nova Dynamics transcription successful!")
        except Exception as e:
            logger.warning(f"Nova Dynamics failed: {e}. Trying other models...")
        
        # PRIORITY 2: Try other cloud APIs if Nova failed
        if segments is None and groq_api_key:
            try:
                logger.info("Trying Groq Cloud API...")
                segments, detected_language = transcribe_via_groq(temp_file_path, groq_api_key, language)
                model_used = "groq-cloud"
                logger.info("Groq transcription successful!")
            except Exception as e:
                logger.warning(f"Groq failed: {e}")
                
        if segments is None and gemini_api_key:
            try:
                logger.info("Trying Gemini Cloud API...")
                segments, detected_language = transcribe_via_gemini(temp_file_path, gemini_api_key)
                model_used = "gemini-cloud"
                logger.info("Gemini transcription successful!")
            except Exception as e:
                logger.warning(f"Gemini failed: {e}")
        
        # PRIORITY 2: Use Model Pool (automatic round-robin distribution)
        if segments is None and model_pool.ready:
            try:
                logger.info("Using Model Pool (auto-distribution)...")
                segments, detected_language, model_used = model_pool.transcribe(temp_file_path, language)
                logger.info(f"Model Pool transcription successful! Used: {model_used}")
            except Exception as e:
                logger.warning(f"Model Pool failed: {e}")
                
        # PRIORITY 3: Fallback to basic Whisper if pool not ready
        if segments is None:
            logger.info("Fallback: Loading basic Whisper tiny model...")
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
                logger.error(f"All transcription methods failed: {e}")
                raise Exception("No transcription model available")
        
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
            
        logger.info(f"Transcription completed. Model: {model_used}, Language: {detected_language}")
        
        return {
            "success": True,
            "detected_language": detected_language,
            "language_probability": language_probability,
            "segments": result_segments,
            "full_text": " ".join(full_text_list),
            "model_used": model_used
        }
        
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/save")
async def save_transcription(request: SaveRequest):
    """Saves the transcription to Word (.docx) and generates a PDF with summaries."""
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
            
        logger.info(f"Saving transcription for session: {session_id}...")
        
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
        logger.error(f"Error saving files: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/sessions")
async def get_sessions():
    """Lists all previous saved sessions."""
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
    """Downloads the requested document."""
    if file_type == "docx":
        filename = f"{session_id}.docx"
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif file_type == "pdf":
        filename = f"{session_id}_summary.pdf"
        media_type = "application/pdf"
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Use 'docx' or 'pdf'.")
        
    file_path = os.path.join(TRANSCRIPTION_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {filename} not found.")
        
    return FileResponse(path=file_path, filename=filename, media_type=media_type)

@app.get("/")
async def root():
    """Redirect root to landing page."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/landing.html")

@app.get("/app")
async def app_redirect():
    """Redirect /app to main transcription workspace."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/index.html")

@app.get("/download-apk")
async def download_apk():
    """Download the Android APK file."""
    apk_path = os.path.join(os.path.dirname(__file__), "app-debug.apk")
    if not os.path.exists(apk_path):
        raise HTTPException(status_code=404, detail="APK file not found")
    return FileResponse(
        path=apk_path,
        filename="AuraScribe-Pro.apk",
        media_type="application/vnd.android.package-archive"
    )

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
