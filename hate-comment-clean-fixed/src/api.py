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

# Load model on startup
model = None
yt_service = None

@app.on_event("startup")
def load_resources():
    global model, yt_service
    # Only load if not already loaded (useful for reload)
    if model is None:
        model = HateCommentClassifier()
    if yt_service is None:
        yt_service = YouTubeService()

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

class AnalyzeVideoResponse(BaseModel):
    video_url: str
    total_comments: int
    toxic_count: int
    safe_count: int
    comments: List[CommentAnalysis]

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

@app.post("/analyze-video", response_model=AnalyzeVideoResponse)
def analyze_video(request: AnalyzeVideoRequest):
    if not model or not yt_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # 1. Fetch Comments
        raw_comments = yt_service.fetch_comments(request.url, limit=request.limit)
        if not raw_comments:
            return {
                "video_url": request.url,
                "total_comments": 0,
                "toxic_count": 0,
                "safe_count": 0,
                "comments": []
            }
        
        # 2. Extract texts for batch prediction
        texts = [c['text'] for c in raw_comments]
        
        # 3. Batch Predict
        predictions = model.predict_batch(texts)
        
        # 4. Merge results
        analyzed_comments = []
        toxic_count = 0
        safe_count = 0
        
        for i, pred in enumerate(predictions):
            label = pred['label']
            # Heuristic: Hate Speech or Offensive Language = Toxic
            is_toxic = label in ["Hate Speech", "Offensive Language"]
            
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
            
        return {
            "video_url": request.url,
            "total_comments": len(analyzed_comments),
            "toxic_count": toxic_count,
            "safe_count": safe_count,
            "comments": analyzed_comments
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error analyzing video: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
