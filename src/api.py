import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
try:
    import hf_transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
import shutil
import uvicorn
import torch
from transformers import pipeline
from src.inference import HateCommentClassifier
from src.youtube_service import YouTubeService
from src.audio_service import AudioService

app = FastAPI(
    title="HateGuard API",
    description="API for detecting toxicity in YouTube comments and Offline Media.",
    version="1.1.0"
)

# Enable CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Wildcard origins (*) cannot be used with credentials=True
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy loading resources
model = None
yt_service = None
audio_service = None
translator = None

def get_model():
    global model
    if model is None:
        model = HateCommentClassifier()
    return model

def get_yt_service():
    global yt_service
    if yt_service is None:
        yt_service = YouTubeService()
    return yt_service

def get_audio_service():
    global audio_service
    if audio_service is None:
        audio_service = AudioService()
    return audio_service

def get_translator():
    global translator
    if translator is None:
        print("\n--- [INIT] Loading Multilingual-to-English translator (one-time) ---")
        try:
            # Task name for these models is often 'translation' or 'text2text-generation'
            # 'text2text-generation' is the most compatible name in current transformers versions
            translator = pipeline("text2text-generation", model="Helsinki-NLP/opus-mt-mul-en", device=(0 if torch.cuda.is_available() else -1))
            print("--- [SUCCESS] Translator model loaded on GPU/CPU. ---\n")
        except Exception as e:
            print(f"--- [ERROR] Translator failed to load: {e}. Translation will be skipped. ---")
            translator = False # Prevent retrying every time
    return translator

def translate_if_needed(text):
    """
    Optional: Translate text to English for display convenience.
    However, our classifier now handles multilingual text natively.
    """
    if not text or not isinstance(text, str):
        return ""
        
    # Only translate if clearly not English and long enough to be a sentence
    if any(ord(char) > 127 for char in text) and len(text) > 10:
        try:
            trans = get_translator()
            if trans: # Only try if translator loaded successfully
                result = trans(text, max_length=128)
                return f"{text} (EN: {result[0]['translation_text']})"
        except Exception:
            pass
    return text

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict

class AnalyzeVideoRequest(BaseModel):
    url: str
    limit: Optional[int] = 100

class AnalysisItem(BaseModel):
    text: str
    author: str
    label: str
    confidence: float
    start: Optional[float] = None
    end: Optional[float] = None

class AnalyzeResponse(BaseModel):
    source: str
    total_segments: int
    toxic_count: int
    safe_count: int
    analysis: List[AnalysisItem]

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    classifier = get_model()
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return classifier.predict(request.text)

@app.post("/analyze-video", response_model=AnalyzeResponse)
def analyze_video(request: AnalyzeVideoRequest):
    classifier = get_model()
    yt = get_yt_service()
    
    print(f"\n--- Starting Analysis for: {request.url} ---")
    try:
        # 1. Fetch Comments
        print("[Step 1/4] Fetching YouTube comments...")
        try:
            raw_comments = yt.fetch_comments(request.url, limit=request.limit)
        except Exception as e:
            print(f"Warning: Comment fetch failed: {e}")
            raw_comments = []
            
        # 2. Extract Audio Transcription
        print("[Step 2/4] Extracting and Transcribing Video Audio (this may take 1-3 mins)...")
        try:
            audio_chunks_raw = yt.extract_video_content(request.url)
            print(f"DEBUG: Successfully extracted {len(audio_chunks_raw)} speech segments.")
        except Exception as e:
            print(f"CRITICAL: Audio transcription failed. Error: {e}")
            import traceback
            traceback.print_exc()
            audio_chunks_raw = []

        # 3. Combine for batch processing
        print("[Step 3/4] Running toxicity detection on all content...")
        texts = [c['text'] for c in raw_comments] + [chunk['text'] for chunk in audio_chunks_raw]
        
        if not texts:
            return {
                "source": request.url,
                "total_segments": 0,
                "toxic_count": 0,
                "safe_count": 0,
                "analysis": []
            }

        predictions = classifier.predict_batch(texts)
        
        # 4. Merge results
        print("[Step 4/4] Finalizing report and translating to English...")
        analyzed_items = []
        toxic_count = 0
        
        # Process Comments
        for i, pred in enumerate(predictions[:len(raw_comments)]):
            is_toxic = pred['label'] in ["Hate Speech", "Offensive Language", "Abusive"]
            if is_toxic: toxic_count += 1
            
            # Translate if needed
            display_text = translate_if_needed(raw_comments[i]['text'])
            
            analyzed_items.append({
                "text": display_text,
                "author": raw_comments[i]['author'],
                "label": pred['label'],
                "confidence": pred['confidence']
            })
            
        # Process Audio
        for i, pred in enumerate(predictions[len(raw_comments):]):
            idx = i
            is_toxic = pred['label'] in ["Hate Speech", "Offensive Language", "Abusive"]
            if is_toxic: toxic_count += 1
            analyzed_items.append({
                "text": audio_chunks_raw[idx]['text'],
                "author": "Speaker (Video)",
                "label": pred['label'],
                "confidence": pred['confidence'],
                "start": audio_chunks_raw[idx]['timestamp'][0],
                "end": audio_chunks_raw[idx]['timestamp'][1]
            })

        print(f"--- Analysis Complete! Flagged {toxic_count} toxic items. ---\n")
        return {
            "source": request.url,
            "total_segments": len(analyzed_items),
            "toxic_count": toxic_count,
            "safe_count": len(analyzed_items) - toxic_count,
            "analysis": analyzed_items
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-file", response_model=AnalyzeResponse)
async def analyze_file(file: UploadFile = File(...)):
    classifier = get_model()
    service = get_audio_service()

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        transcription = service.transcribe(temp_path)
        segments = transcription['segments']
        
        if not segments:
            return {
                "source": file.filename,
                "total_segments": 0,
                "toxic_count": 0,
                "safe_count": 0,
                "analysis": []
            }

        texts = [s['text'] for s in segments]
        predictions = classifier.predict_batch(texts)
        
        analyzed_items = []
        toxic_count = 0
        
        for i, pred in enumerate(predictions):
            is_toxic = pred['label'] in ["Hate Speech", "Offensive Language", "Abusive"]
            if is_toxic: toxic_count += 1
            analyzed_items.append({
                "text": segments[i]['text'],
                "author": "Speaker",
                "label": pred['label'],
                "confidence": pred['confidence'],
                "start": segments[i]['start'],
                "end": segments[i]['end']
            })
            
        return {
            "source": file.filename,
            "total_segments": len(analyzed_items),
            "toxic_count": toxic_count,
            "safe_count": len(analyzed_items) - toxic_count,
            "analysis": analyzed_items
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# Mount the pre-built UI if it exists
if os.path.exists("ui/dist"):
    print("Mounting pre-built UI from ui/dist")
    app.mount("/", StaticFiles(directory="ui/dist", html=True), name="ui")

def clear_port(port=8000):
    import platform
    import subprocess
    if platform.system() != "Windows":
        print(f"Clearing any process occupying port {port}...")
        try:
            subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)
        except Exception as e:
            print(f"Failed to clear port: {e}")

if __name__ == "__main__":
    clear_port(8000)
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
