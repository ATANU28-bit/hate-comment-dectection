import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
import shutil
from src.inference import HateCommentClassifier
from src.youtube_service import YouTubeService
from src.audio_service import AudioService

app = FastAPI(
    title="Hate Comment Detection API",
    description="API for classifying text and YouTube video comments as Hate Speech, Offensive Language, or Neither.",
    version="1.0.0"
)

# Enable CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only. In prod, restrict to specific domains.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load resources on startup
model = None
yt_service = None
audio_service = None

@app.on_event("startup")
def load_resources():
    global model, yt_service, audio_service
    if model is None:
        model = HateCommentClassifier()
    if yt_service is None:
        yt_service = YouTubeService()
    if audio_service is None:
        audio_service = AudioService()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict

class AnalyzeVideoRequest(BaseModel):
    url: str
    limit: Optional[int] = 50

class CommentAnalysis(BaseModel):
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
    analysis: List[CommentAnalysis]

@app.get("/")
def read_root():
    return {"message": "Hate Comment Detection API is running. Go to /docs for Swagger UI."}

@app.get("/health")
def health_check():
    status = "healthy" if model else "unhealthy"
    return {"status": status, "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    result = model.predict(request.text)
    return result

@app.post("/analyze-video", response_model=AnalyzeResponse)
def analyze_video(request: AnalyzeVideoRequest):
    if not model or not yt_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        raw_comments = yt_service.fetch_comments(request.url, limit=request.limit)
        if not raw_comments:
            return {
                "source": request.url,
                "total_segments": 0,
                "toxic_count": 0,
                "safe_count": 0,
                "analysis": []
            }
        
        texts = [c['text'] for c in raw_comments]
        predictions = model.predict_batch(texts)
        
        analyzed_items = []
        toxic_count = 0
        safe_count = 0
        
        for i, pred in enumerate(predictions):
            label = pred['label']
            is_toxic = label in ["Hate Speech", "Offensive Language"]
            
            if is_toxic:
                toxic_count += 1
            else:
                safe_count += 1
                
            analyzed_items.append({
                "text": raw_comments[i]['text'],
                "author": raw_comments[i]['author'],
                "label": label,
                "confidence": pred['confidence']
            })
            
        return {
            "source": request.url,
            "total_segments": len(analyzed_items),
            "toxic_count": toxic_count,
            "safe_count": safe_count,
            "analysis": analyzed_items
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error analyzing video: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/analyze-file", response_model=AnalyzeResponse)
async def analyze_file(file: UploadFile = File(...)):
    if not model or not audio_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # 1. Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Transcribe
        transcription = audio_service.transcribe(temp_path)
        segments = transcription['segments']
        
        if not segments:
            return {
                "source": file.filename,
                "total_segments": 0,
                "toxic_count": 0,
                "safe_count": 0,
                "analysis": []
            }

        # 3. Batch Predict
        texts = [s['text'] for s in segments]
        predictions = model.predict_batch(texts)
        
        # 4. Merge results
        analyzed_items = []
        toxic_count = 0
        safe_count = 0
        
        for i, pred in enumerate(predictions):
            label = pred['label']
            is_toxic = label in ["Hate Speech", "Offensive Language"]
            
            if is_toxic:
                toxic_count += 1
            else:
                safe_count += 1
                
            analyzed_items.append({
                "text": segments[i]['text'],
                "author": "Speaker", # In audio/video, we don't know the author easily
                "label": label,
                "confidence": pred['confidence'],
                "start": segments[i]['start'],
                "end": segments[i]['end']
            })
            
        return {
            "source": file.filename,
            "total_segments": len(analyzed_items),
            "toxic_count": toxic_count,
            "safe_count": safe_count,
            "analysis": analyzed_items
        }

    except Exception as e:
        print(f"Error analyzing file: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
