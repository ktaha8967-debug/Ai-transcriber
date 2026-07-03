import os
import shutil
import uuid
import logging
from typing import List, Optional, Any
from datetime import datetime
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel

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

app = FastAPI(title="Voice Transcriber & Summarizer")

# Session structure to hold in-memory drafts
class SaveRequest(BaseModel):
    session_id: str
    text_segments: List[Any]
    language: Optional[str] = "en"

# Cache for Whisper models to prevent reloading
models_cache = {}

def get_whisper_model(model_size: str = "tiny"):
    """Loads and caches the Whisper model."""
    if model_size not in models_cache:
        logger.info(f"Loading Whisper model: {model_size}...")
        try:
            # CPU INT8 is fast, memory-efficient, and requires no CUDA
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            models_cache[model_size] = model
            logger.info(f"Whisper model {model_size} loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model {model_size}: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    return models_cache[model_size]

def transcribe_via_groq(file_path: str, api_key: str, model_size: str = "whisper-large-v3", language: str = "auto"):
    """Translates audio file to English using Groq Whisper API."""
    url = "https://api.groq.com/openai/v1/audio/translations"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    # Map model sizes to Groq models
    groq_model = "whisper-large-v3"
    if model_size in ["tiny", "base"]:
        groq_model = "whisper-large-v3-turbo"
    else:
        groq_model = "whisper-large-v3"
        
    files = {
        "file": (os.path.basename(file_path), open(file_path, "rb"), "audio/wav")
    }
    data = {
        "model": groq_model,
        "response_format": "verbose_json"
    }
        
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

def transcribe_via_kimi(file_path: str, api_key: str, language: str = "auto"):
    """Translates audio file to English using Kimi (Moonshot AI) Audio Translation API."""
    url = "https://api.moonshot.cn/v1/audio/translations"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    files = {
        "file": (os.path.basename(file_path), open(file_path, "rb"), "audio/wav")
    }
    data = {
        "model": "whisper-1",
        "response_format": "verbose_json"
    }
        
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, headers=headers, files=files, data=data)
            if response.status_code != 200:
                raise Exception(f"Kimi API error: {response.text}")
            
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
    """Translates audio file to English using Google Gemini API."""
    import base64
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    with open(file_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode("utf-8")
        
    payload = {
        "contents": [{
            "parts": [
                {
                    "inlineData": {
                        "mimeType": "audio/wav",
                        "data": audio_data
                    }
                },
                {
                    "text": (
                        "Translate the spoken audio content into English. Return the result strictly as a JSON array of segment objects. "
                        "Each object must contain three keys: 'start' (float, start time in seconds), "
                        "'end' (float, end time in seconds), and 'text' (string, translated English text). "
                        "Do not include any markdown formatting (like ```json), return only raw JSON."
                    )
                }
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
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

def transcribe_via_openai(file_path: str, api_key: str, language: str = "auto"):
    """Translates audio file to English using OpenAI Whisper API."""
    url = "https://api.openai.com/v1/audio/translations"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    files = {
        "file": (os.path.basename(file_path), open(file_path, "rb"), "audio/wav")
    }
    data = {
        "model": "whisper-1",
        "response_format": "verbose_json"
    }
        
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, headers=headers, files=files, data=data)
            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.text}")
            
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

# Ensure transcriptions directory exists
TRANSCRIPTION_DIR = "transcriptions"
if not os.path.exists(TRANSCRIPTION_DIR):
    os.makedirs(TRANSCRIPTION_DIR)

# Static files directory setup
STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Pre-load the base model on startup in a separate thread/task
@app.on_event("startup")
async def startup_event():
    try:
        # Load the default 'base' model so it is cached and ready to use
        get_whisper_model("base")
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e}. It will load on first request.")

@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    model_size: str = Form("tiny"),
    language: str = Form("auto")
):
    """
    Transcribes the uploaded audio file.
    """
    # Dynamically load API keys from env file if available
    load_env()
    kimi_api_key = os.environ.get("KIMI_API_KEY")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    # Create a temporary file to save the uploaded audio
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
    
    segments = None
    detected_language = "auto"
    language_probability = 1.0
    
    global KIMI_AVAILABLE
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Saved temporary file to {temp_file_path}. Transcribing with model {model_size}...")
        
        # 1. Try Kimi first if key is available and Kimi is marked as available
        if kimi_api_key and KIMI_AVAILABLE:
            try:
                logger.info("KIMI_API_KEY found. Transcribing using Kimi (Moonshot AI) Cloud API...")
                segments, detected_language = transcribe_via_kimi(temp_file_path, kimi_api_key, language)
                logger.info("Transcription via Kimi successful!")
            except httpx.RequestError as e:
                logger.warning(f"Kimi network/DNS connection error: {e}. Disabling Kimi for this session to avoid DNS timeout delays. Trying fallback...")
                KIMI_AVAILABLE = False
            except Exception as e:
                logger.warning(f"Kimi transcription failed: {e}. Trying fallback...")
                
        # 2. Try Groq if Kimi failed or was not available
        if segments is None and groq_api_key:
            try:
                logger.info("GROQ_API_KEY found. Transcribing using Groq Cloud API...")
                segments, detected_language = transcribe_via_groq(temp_file_path, groq_api_key, model_size, language)
                logger.info("Transcription via Groq successful!")
            except Exception as e:
                logger.warning(f"Groq transcription failed: {e}. Trying fallback...")
                
        # 3. Try Gemini if Groq failed or was not available
        if segments is None and gemini_api_key:
            try:
                logger.info("GEMINI_API_KEY found. Transcribing using Gemini 2.5 Flash API...")
                segments, detected_language = transcribe_via_gemini(temp_file_path, gemini_api_key)
                logger.info("Transcription via Gemini successful!")
            except Exception as e:
                logger.warning(f"Gemini transcription failed: {e}. Trying fallback...")
                
        # 4. Try OpenAI if Gemini failed or was not available
        if segments is None and openai_api_key:
            try:
                logger.info("OPENAI_API_KEY found. Transcribing using OpenAI Whisper API...")
                segments, detected_language = transcribe_via_openai(temp_file_path, openai_api_key, language)
                logger.info("Transcription via OpenAI successful!")
            except Exception as e:
                logger.warning(f"OpenAI transcription failed: {e}. Trying local fallback...")
                
        # 5. Fallback to local Whisper CPU if all cloud options failed/unavailable (optimized for speed)
        if segments is None:
            logger.info("No cloud API keys succeeded. Falling back to local Whisper...")
            # Load whisper model
            model = get_whisper_model(model_size)
            
            # Set transcription params (use beam_size=1 for fast fallback on CPU, task=translate to translate to English)
            transcribe_args = {
                "beam_size": 1,
                "vad_filter": True, # Voice Activity Detection filters out silence
                "task": "translate"
            }
            if language != "auto":
                transcribe_args["language"] = language
                
            local_segments, info = model.transcribe(temp_file_path, **transcribe_args)
            
            segments = []
            for seg in local_segments:
                segments.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text
                })
            detected_language = info.language
            language_probability = info.language_probability
            
        # Collect transcription segments
        result_segments = []
        full_text_list = []
        for segment in segments:
            # Format timestamp as MM:SS
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
            
        logger.info(f"Transcription completed. Detected language: {detected_language}")
        
        return {
            "success": True,
            "detected_language": detected_language,
            "language_probability": language_probability,
            "segments": result_segments,
            "full_text": " ".join(full_text_list)
        }
        
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/save")
async def save_transcription(request: SaveRequest):
    """
    Saves the transcription to Word (.docx) and generates a PDF with summaries.
    Autosaves immediately on call.
    """
    try:
        session_id = request.session_id
        text_segments = request.text_segments
        
        # Combine text segments into a single full text block for summarization
        full_text = " ".join([seg.get("text", "") if isinstance(seg, dict) else str(seg) for seg in text_segments])
        
        # Summarize if we have content
        summary_points = []
        if len(full_text.strip()) > 30:
            summary, bullet_points = summarizer.summarize_text(full_text, num_sentences=4)
            summary_points = bullet_points
        else:
            summary_points = ["Audio transcription was too short to generate a summary."]
            
        logger.info(f"Saving transcription for session: {session_id}...")
        
        # Generate DOCX and PDF documents
        docx_path = document_generator.save_to_docx(session_id, text_segments, summary_points)
        pdf_path = document_generator.save_to_pdf(session_id, summary_points, full_text)
        
        # Save a text draft just in case
        draft_path = os.path.join(TRANSCRIPTION_DIR, f"{session_id}_draft.txt")
        
        # Format the draft file with line breaks for segments
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
    """
    Lists all previous saved sessions.
    """
    try:
        files = os.listdir(TRANSCRIPTION_DIR)
        # Find unique session IDs by checking .docx files
        sessions = []
        for file in files:
            if file.endswith(".docx"):
                session_id = file[:-5]
                docx_path = os.path.join(TRANSCRIPTION_DIR, file)
                pdf_file = f"{session_id}_summary.pdf"
                pdf_exists = pdf_file in files
                
                # Get file modification time
                mtime = os.path.getmtime(docx_path)
                formatted_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                # Try to read a preview
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
                
        # Sort sessions by most recent modification time
        sessions.sort(key=lambda x: x["mtime"], reverse=True)
        return {"success": True, "sessions": sessions}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/download/{file_type}/{session_id}")
async def download_file(file_type: str, session_id: str):
    """
    Downloads the requested document.
    """
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

# Serve the static files (Frontend)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8000
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
