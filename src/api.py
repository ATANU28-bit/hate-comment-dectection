import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from src.inference import HateCommentClassifier
from src.youtube_service import YouTubeService

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

# Load model lazily to allow instant server startup
model = None
yt_service = None

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

class AudioAnalysis(BaseModel):
    text: str
    timestamp: tuple
    label: str
    confidence: float

class AnalyzeVideoResponse(BaseModel):
    video_url: str
    total_comments: int
    toxic_count: int
    safe_count: int
    comments: List[CommentAnalysis]
    audio_chunks: List[AudioAnalysis] = []
    toxic_audio_count: int = 0

@app.get("/")
def read_root():
    return {"message": "Hate Comment Detection API is running. Go to /docs for Swagger UI."}

@app.get("/health")
def health_check():
    status = "healthy" if model else "unhealthy"
    return {"status": status, "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    classifier = get_model()
    if not classifier:
        raise HTTPException(status_code=503, detail="Model failed to load")
    
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    result = classifier.predict(request.text)
    return result

@app.post("/analyze-video", response_model=AnalyzeVideoResponse)
def analyze_video(request: AnalyzeVideoRequest):
    classifier = get_model()
    service = get_yt_service()
    if not classifier or not service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # 1. Fetch Comments and Audio Chunks
        raw_comments = service.fetch_comments(request.url, limit=request.limit)
        audio_chunks_raw = service.extract_video_content(request.url)
        
        # 2. Extract texts for batch prediction
        comment_texts = [c['text'] for c in raw_comments]
        audio_texts = [chunk['text'] for chunk in audio_chunks_raw]
        
        # 3. Batch Predict
        all_texts = comment_texts + audio_texts
        if not all_texts:
            return {
                "video_url": request.url,
                "total_comments": 0,
                "toxic_count": 0,
                "safe_count": 0,
                "comments": [],
                "audio_chunks": [],
                "toxic_audio_count": 0
            }
            
        predictions = classifier.predict_batch(all_texts)
        
        # Split predictions
        comment_preds = predictions[:len(comment_texts)]
        audio_preds = predictions[len(comment_texts):]
        
        # 4. Merge results for comments
        analyzed_comments = []
        toxic_count = 0
        safe_count = 0
        
        for i, pred in enumerate(comment_preds):
            label = pred['label']
            is_toxic = label == "Abusive"
            
            if is_toxic:
                toxic_count += 1
            else:
                safe_count += 1
                
            analyzed_comments.append({
                "text": raw_comments[i]['text'],
                "author": raw_comments[i]['author'],
                "label": label,
                "confidence": pred['confidence']
            })
            
        # 5. Merge results for audio chunks
        analyzed_audio = []
        toxic_audio_count = 0
        
        for i, pred in enumerate(audio_preds):
            label = pred['label']
            is_toxic = label == "Abusive"
            
            if is_toxic:
                toxic_audio_count += 1
                
            analyzed_audio.append({
                "text": audio_chunks_raw[i]['text'],
                "timestamp": audio_chunks_raw[i]['timestamp'],
                "label": label,
                "confidence": pred['confidence']
            })
            
        return {
            "video_url": request.url,
            "total_comments": len(analyzed_comments),
            "toxic_count": toxic_count,
            "safe_count": safe_count,
            "comments": analyzed_comments,
            "audio_chunks": analyzed_audio,
            "toxic_audio_count": toxic_audio_count
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error analyzing video: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
