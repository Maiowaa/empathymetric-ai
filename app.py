from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from fastapi.responses import FileResponse, JSONResponse
from transformers import pipeline
import torch
import os
from datetime import datetime
from typing import Optional, List, Dict
import logging
from contextlib import contextmanager
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EmpathyMetric AI",
    description="Advanced emotion classification",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "./model")
DB_PATH = os.getenv("DB_PATH", "empathy_history.db")

id2label = {
    0: "sadness", 
    1: "joy", 
    2: "love", 
    3: "anger", 
    4: "fear", 
    5: "surprise"
}

emotion_colors = {
    "sadness": "#6C88C4",
    "joy": "#FFD93D",
    "love": "#FF6B9D",
    "anger": "#E63946",
    "fear": "#9D4EDD",
    "surprise": "#06FFA5"
}

emotion_descriptions = {
    "sadness": "Feelings of sorrow, grief, or melancholy",
    "joy": "Positive emotions of happiness and delight",
    "love": "Affection, care, and warmth towards others",
    "anger": "Frustration, irritation, or hostility",
    "fear": "Anxiety, worry, or apprehension",
    "surprise": "Astonishment or unexpected reactions"
}

class EmotionClassifier:
    """
    Maps 7-emotion pre-trained model to 6 target emotions
    """
    def __init__(self, model_path):
        # Verify model directory exists
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model directory not found: {model_path}")
        
        # Check for required model files
        required_files = ['config.json', 'tokenizer.json']
        missing_files = []
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            raise RuntimeError(f"Missing required model files: {missing_files}. Check build logs.")
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Model files found: {os.listdir(model_path)}")
        
        try:
            self.classifier = pipeline(
                "text-classification",
                model=model_path,
                tokenizer=model_path,
                device=-1,  # Force CPU (Render free tier)
                top_k=None,
                local_files_only=True  # Don't try to download from HuggingFace
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
        
        self.emotion_map = {
            "anger": "anger",
            "disgust": "anger",
            "fear": "fear",
            "joy": "joy",
            "neutral": "love",
            "sadness": "sadness",
            "surprise": "surprise"
        }
        
        self.labels = {
            "sadness": 0,
            "joy": 1,
            "love": 2,
            "anger": 3,
            "fear": 4,
            "surprise": 5
        }
        
        logger.info("✅ Model loaded successfully")
    
    def predict(self, text: str) -> Dict:
        """Predict emotion from text"""
        results = self.classifier(text)[0]
        emotion_scores = {e: 0.0 for e in self.labels.keys()}
        
        for result in results:
            original_emotion = result['label']
            score = result['score']
            mapped_emotion = self.emotion_map.get(original_emotion, "love")
            emotion_scores[mapped_emotion] += score
        
        top_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        return {
            "label": top_emotion[0],
            "confidence": top_emotion[1],
            "confidence_percentage": f"{top_emotion[1]*100:.1f}%",
            "category_id": self.labels[top_emotion[0]],
            "all_probabilities": {k: round(v*100, 1) for k, v in emotion_scores.items()}
        }

# MODEL LOADING
try:
    logger.info(f"Initializing EmpathyMetric AI...")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Absolute path: {os.path.abspath(MODEL_PATH)}")
    
    classifier = EmotionClassifier(MODEL_PATH)
    device = "cpu"
    logger.info(f"✅ Model loaded on {device}")
    logger.info("✅ EmpathyMetric Engine Ready [Pre-trained Model, 6 emotions]")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    logger.error(f"Current directory: {os.getcwd()}")
    logger.error(f"Directory contents: {os.listdir('.')}")
    if os.path.exists(MODEL_PATH):
        logger.error(f"Model directory contents: {os.listdir(MODEL_PATH)}")
    raise RuntimeError(f"Failed to load model: {e}")

# DATABASE SETUP
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  text TEXT NOT NULL,
                  label TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_timestamp 
                 ON history(timestamp DESC)''')
    conn.commit()
    conn.close()
    logger.info("✅ Database initialized")

init_db()

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

# REQUEST/RESPONSE MODELS
class EmpathyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

# API ROUTES
@app.get("/")
async def serve_home():
    """Serve main HTML interface"""
    html_path = "templates/index.html"
    if not os.path.exists(html_path):
        return JSONResponse(
            content={
                "message": "EmpathyMetric AI API",
                "version": "2.0",
                "docs": "/docs",
                "status": "healthy"
            }
        )
    return FileResponse(html_path)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "device": str(device),
        "emotions": list(id2label.values()),
        "model_type": "pre-trained (emotion-english-distilroberta-base)"
    }

@app.post("/predict")
async def predict_empathy(request: EmpathyRequest):
    """Predict emotion from text"""
    start_time = datetime.now()
    
    try:
        text = request.text
        
        # Get prediction using wrapper
        result = classifier.predict(text)
        
        label = result['label']
        confidence_score = result['confidence']
        all_probs = result['all_probabilities']
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Save to database
        try:
            with get_db() as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO history (text, label, confidence) VALUES (?, ?, ?)",
                    (text, label, confidence_score)
                )
                conn.commit()
        except Exception as db_error:
            logger.warning(f"Database save failed: {db_error}")
        
        logger.info(f"Prediction: {label} ({confidence_score:.2%}) for: '{text[:50]}...'")
        
        return {
            "label": label,
            "confidence": confidence_score,
            "confidence_percentage": result['confidence_percentage'],
            "category_id": result['category_id'],
            "all_probabilities": all_probs,
            "color": emotion_colors.get(label, "#888888"),
            "description": emotion_descriptions.get(label, ""),
            "processing_time": round(processing_time, 2)
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/history")
async def get_history(
    limit: int = Query(10, ge=1, le=100),
    emotion: Optional[str] = Query(None)
):
    """Get prediction history"""
    try:
        with get_db() as conn:
            c = conn.cursor()
            
            if emotion and emotion in id2label.values():
                query = """SELECT id, text, label, confidence, timestamp 
                          FROM history WHERE label = ? 
                          ORDER BY timestamp DESC LIMIT ?"""
                c.execute(query, (emotion, limit))
            else:
                query = """SELECT id, text, label, confidence, timestamp 
                          FROM history ORDER BY timestamp DESC LIMIT ?"""
                c.execute(query, (limit,))
            
            rows = c.fetchall()
        
        return [
            {
                "id": r[0],
                "text": r[1],
                "label": r[2],
                "confidence": round(r[3] * 100, 2),
                "timestamp": r[4],
                "color": emotion_colors.get(r[2], "#888888")
            }
            for r in rows
        ]
        
    except Exception as e:
        logger.error(f"History error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_statistics():
    """Get statistics"""
    try:
        with get_db() as conn:
            c = conn.cursor()
            
            c.execute("SELECT COUNT(*) FROM history")
            total = c.fetchone()[0]
            
            c.execute("SELECT label, COUNT(*) FROM history GROUP BY label")
            distribution = dict(c.fetchall())
            
            c.execute("SELECT AVG(confidence) FROM history")
            avg_conf = c.fetchone()[0] or 0
            
            most_common = max(distribution.items(), key=lambda x: x[1])[0] if distribution else "None"
        
        return {
            "total_predictions": total,
            "emotion_distribution": distribution,
            "average_confidence": round(avg_conf * 100, 2),
            "most_common_emotion": most_common
        }
        
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history")
async def clear_history():
    """Clear all history"""
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM history")
            conn.commit()
            deleted_count = c.rowcount
        
        logger.info(f"Cleared {deleted_count} records")
        return {"message": f"Deleted {deleted_count} records"}
        
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/emotions")
async def get_emotions():
    """Get all emotions"""
    return {
        "emotions": [
            {
                "id": k,
                "name": v,
                "color": emotion_colors.get(v, "#888888"),
                "description": emotion_descriptions.get(v, "")
            }
            for k, v in id2label.items()
        ]
    }

@app.post("/batch-predict")
async def batch_predict(texts: List[str] = Field(..., max_items=50)):
    """Batch prediction"""
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    try:
        results = []
        for text in texts:
            request = EmpathyRequest(text=text)
            result = await predict_empathy(request)
            results.append(result)
        
        return {"predictions": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 EmpathyMetric API Starting...")
    logger.info(f"📊 Tracking {len(id2label)} emotions: {', '.join(id2label.values())}")
    logger.info("💡 Using pre-trained emotion model (no fine-tuning)")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 EmpathyMetric API Shutting Down...")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
