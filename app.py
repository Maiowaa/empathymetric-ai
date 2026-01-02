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

MODEL_PATH = os.path.abspath("./model")
DB_PATH = "empathy_history.db"

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
    def __init__(self, model_path):
        self.classifier = pipeline(
            "text-classification",
            model=model_path,
            device=0 if torch.cuda.is_available() else -1,
            top_k=None
	    local_files_only=True
        )
        
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
            "sadness": 0, "joy": 1, "love": 2,
            "anger": 3, "fear": 4, "surprise": 5
        }
    
    def predict(self, text: str) -> Dict:
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

try:
    logger.info(f"Loading model from: {MODEL_PATH}")
    classifier = EmotionClassifier(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"✅ Model loaded on {device}")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    raise

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  text TEXT NOT NULL,
                  label TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

class EmpathyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

@app.get("/")
async def serve_home():
    return FileResponse("templates/index.html")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device),
        "emotions": list(id2label.values())
    }

@app.post("/predict")
async def predict_empathy(request: EmpathyRequest):
    start_time = datetime.now()
    
    try:
        result = classifier.predict(request.text)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        with get_db() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO history (text, label, confidence) VALUES (?, ?, ?)",
                (request.text, result['label'], result['confidence'])
            )
            conn.commit()
        
        return {
            **result,
            "color": emotion_colors.get(result['label'], "#888888"),
            "description": emotion_descriptions.get(result['label'], ""),
            "processing_time": round(processing_time, 2)
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history(limit: int = Query(10, ge=1, le=100)):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT text, label, confidence, timestamp FROM history ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = c.fetchall()
    
    return [
        {
            "text": r[0],
            "label": r[1],
            "confidence": round(r[2] * 100, 1),
            "timestamp": r[3],
            "color": emotion_colors.get(r[1], "#888888")
        }
        for r in rows
    ]

@app.get("/stats")
async def get_statistics():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM history")
        total = c.fetchone()[0]
        c.execute("SELECT label, COUNT(*) FROM history GROUP BY label")
        distribution = dict(c.fetchall())
        c.execute("SELECT AVG(confidence) FROM history")
        avg_conf = c.fetchone()[0] or 0
    
    return {
        "total_predictions": total,
        "emotion_distribution": distribution,
        "average_confidence": round(avg_conf * 100, 2),
        "most_common_emotion": max(distribution.items(), key=lambda x: x[1])[0] if distribution else "None"
    }

@app.get("/emotions")
async def get_emotions():
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
