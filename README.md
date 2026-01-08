# 🎭 EmpathyMetric AI - Advanced Emotion Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.35+-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A production-ready emotion classification system using a **pre-trained RoBERTa model**, capable of detecting 6 distinct emotions in text with **60-70% accuracy**. Features a FastAPI backend, interactive web interface, and requires **no training** - works out of the box!

![Demo Screenshot](docs/demo.gif)

## ✨ Key Highlights

-  **Pre-trained Model**: Uses high-quality emotion model (no training needed!)
-  **Fast**: ~20ms inference time per prediction
-  **6 Emotions**: Sadness, Joy, Love, Anger, Fear, Surprise
-  **Production Ready**: FastAPI backend with proper error handling
-  **Real-time Dashboard**: Track predictions and statistics
-  **Docker Support**: Deploy anywhere in minutes

##  Why This Approach?

Unlike other emotion classifiers that require training on your data, EmpathyMetric AI uses a **pre-trained model** that was trained on high-quality, carefully annotated emotion datasets. This means:

- ✅ **Better accuracy** than training on noisy data
- ✅ **Works immediately** without collecting training data
- ✅ **Consistent results** across different use cases
- ✅ **No GPU needed** for deployment

##  Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 60-70% |
| F1-Score | 0.60-0.65 |
| Inference Time | ~20ms |
| Model Size | ~330MB |
| GPU Required | No (CPU works fine) |

##  Quick Start

### Option 1: Docker (Recommended - 2 minutes)

```bash
# Clone and run
git clone https://github.com/Maiowaa/empathymetric-ai.git
cd empathymetric-ai

# Download model (one time)
pip install gdown
gdown https://drive.google.com/uc?id=1Tz2caKaZTFJ9Tp5-84_Lyxca1YhyDqfF
unzip emotion_model_production.zip -d model/

# Start with Docker
docker-compose up
```

Visit `http://localhost:8000` - Done! 

### Option 2: Local Installation (5 minutes)

```bash
# Clone repository
git clone https://github.com/Maiowaa/empathymetric-ai.git
cd empathymetric-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained model
pip install gdown
gdown https://drive.google.com/uc?id=1Tz2caKaZTFJ9Tp5-84_Lyxca1YhyDqfF
unzip emotion_model_production.zip -d model/

# Run the server
python app.py
```
## Deployment Notes

**Local Development**: Fully functional ✅  
**Free Tier Deployment**: Limited by 512MB RAM ⚠️  
**Recommended**: Paid hosting ($7/month) or local deployment

The model requires ~600MB RAM to run. Free tier hosting services (512MB) are insufficient. For production deployment, use:
- Render Starter Plan ($7/month, 2GB RAM)
- Railway Pro Plan
- AWS/GCP with sufficient resources
- Local deployment with the provided Docker setup

**Live Demo**: Due to hosting costs, no public demo is currently available. Clone and run locally to test!


##  Model Download

The pre-trained model is hosted externally due to size (~330MB):

**Google Drive**: [Download Model](https://drive.google.com/file/d/1Tz2caKaZTFJ9Tp5-84_Lyxca1YhyDqfF/view)

**Or use gdown**:
```bash
pip install gdown
gdown https://drive.google.com/uc?id=1Tz2caKaZTFJ9Tp5-84_Lyxca1YhyDqfF
unzip emotion_model_production.zip -d model/
```

**Or Hugging Face**:
```bash
huggingface-cli download Maiowaa/empathymetric-model --local-dir ./model
```

##  Usage Examples

### Web Interface

Simply open `http://localhost:8000` in your browser and start typing!

### Python SDK

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="./model", top_k=None)

# Analyze emotion
result = classifier("I'm so happy today!")
print(result)
# Output: [{'label': 'joy', 'score': 0.87}, ...]
```

### REST API

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "I'm feeling great!"}
)

result = response.json()
print(f"Emotion: {result['label']}")
print(f"Confidence: {result['confidence_percentage']}")
print(f"All probabilities: {result['all_probabilities']}")
```

**Example Response**:
```json
{
  "label": "joy",
  "confidence": 0.87,
  "confidence_percentage": "87.0%",
  "category_id": 1,
  "all_probabilities": {
    "joy": 87.0,
    "love": 8.5,
    "surprise": 3.2,
    "sadness": 0.8,
    "fear": 0.3,
    "anger": 0.2
  },
  "color": "#FFD93D",
  "description": "Positive emotions of happiness and delight",
  "processing_time": 23.5
}
```

### Batch Processing

```python
# Analyze multiple texts at once
response = requests.post(
    "http://localhost:8000/batch-predict",
    json={"texts": [
        "I love this!",
        "This makes me angry",
        "I'm so scared"
    ]}
)

for pred in response.json()['predictions']:
    print(f"{pred['label']}: {pred['confidence_percentage']}")
```

### Command Line

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "What a wonderful day!"}'

# Get statistics
curl "http://localhost:8000/stats"

# View history
curl "http://localhost:8000/history?limit=10"
```

##  Project Structure

```
empathymetric-ai/
├── app.py                         # FastAPI application
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose
├── test_api.py                    # Comprehensive tests
├── .env.example                   # Config template
├── templates/
│   └── index.html                 # Web interface
├── model/                         # Pre-trained model (download separately)
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
└── docs/
    └── README.md                  # Additional documentation
```

##  Emotion Categories

| Emotion | Description | Color | Example |
|---------|-------------|-------|---------|
| 😢 **Sadness** | Sorrow, grief, melancholy | Blue | "I miss them so much" |
| 😊 **Joy** | Happiness, delight, pleasure | Yellow | "This is amazing!" |
| ❤️ **Love** | Affection, care, warmth | Pink | "You mean the world to me" |
| 😠 **Anger** | Frustration, irritation | Red | "This is unacceptable!" |
| 😰 **Fear** | Anxiety, worry, terror | Purple | "I'm so scared" |
| 😲 **Surprise** | Astonishment, shock | Green | "I never expected this!" |

## 🔧 Configuration

Create a `.env` file:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration  
MODEL_PATH=./model
DEVICE=cpu  # or cuda if you have GPU

# Database
DB_PATH=empathy_history.db

# Logging
LOG_LEVEL=INFO
```

##  API Documentation

Once running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/predict` | POST | Single text prediction |
| `/batch-predict` | POST | Multiple texts |
| `/history` | GET | Prediction history |
| `/stats` | GET | Usage statistics |
| `/emotions` | GET | List all emotions |
| `/health` | GET | Health check |

## 🧪 Testing

```bash
# Run comprehensive tests
python test_api.py

# Expected output:
# ✅ Health Check - PASS
# ✅ Single Prediction - PASS
# ✅ Batch Prediction - PASS
# ✅ History - PASS
# 🎉 All tests passed!
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t empathymetric-ai .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/model:/app/model:ro \
  -v $(pwd)/data:/app/data \
  --name empathymetric \
  empathymetric-ai

# View logs
docker logs -f empathymetric
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

##  Production Deployment

### Performance Optimization

```bash
# Use Gunicorn with multiple workers
gunicorn app:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

### Scaling

- Use **load balancer** (nginx) for multiple instances
- Enable **Redis caching** for frequent predictions
- Deploy on **Kubernetes** for auto-scaling

### Monitoring

- Add **Prometheus** metrics
- Use **Grafana** for dashboards
- Set up **health check endpoints**

## 💻 Model Details

### Base Model

- **Architecture**: DistilRoBERTa (distilled RoBERTa)
- **Pre-training**: emotion-english-distilroberta-base
- **Original Training Data**: High-quality emotion datasets
- **Parameters**: ~82M
- **No Fine-tuning**: Uses pre-trained weights directly

### Emotion Mapping

The model internally predicts 7 emotions which are mapped to our 6:

| Original (7) | Mapped (6) |
|--------------|------------|
| anger | anger |
| disgust | anger |
| fear | fear |
| joy | joy |
| neutral | love |
| sadness | sadness |
| surprise | surprise |

##  Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE)

##  Acknowledgments

- Base model: [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
- FastAPI framework
- Hugging Face Transformers
- The open-source community

##  Contact

**Kushagar** - [GitHub](https://github.com/Maiowaa)


Project Link: [https://github.com/Maiowaa/empathymetric-ai](https://github.com/Maiowaa/empathymetric-ai)

---

##  Roadmap

- [ ] Multi-language support (Spanish, French, German)
- [ ] Emotion intensity scoring (0-100)
- [ ] Real-time streaming API (WebSocket)
- [ ] Mobile SDK (iOS/Android)
- [ ] Browser extension
- [ ] Voice emotion detection
- [ ] Custom emotion categories

##  Star History

If you find this project helpful, please give it a star! ⭐

##  Usage Stats

-  **Active Users**: Coming soon
-  **Predictions Made**: Coming soon
-  **Countries**: Coming soon

---

<div align="center">

**Built using FastAPI and Transformers**

[Report Bug](https://github.com/Maiowaa/empathymetric-ai/issues) · [Request Feature](https://github.com/Maiowaa/empathymetric-ai/issues) · [Documentation](https://github.com/Maiowaa/empathymetric-ai/wiki)

</div>
