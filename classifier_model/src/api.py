import os
import re
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

logging.basicConfig(level=logging.INFO)

# Define request/response schemas
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str

# Initialize FastAPI app
app = FastAPI(title="Political Bias Classifier API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Text cleaning function
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    return text.strip()

# Global model variables
clf = None
vectorizer = None

@app.on_event("startup")
async def load_model():
    global clf, vectorizer

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    model_path = os.path.join(models_dir, "model.pkl")
    vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")

    print(f"Loading model from {model_path}")
    print(f"Loading vectorizer from {vectorizer_path}")

    try:
        clf = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print(f"✅ Model loaded: {clf}")
        print(f"✅ Vectorizer loaded: {vectorizer}")

        np.savetxt(os.path.join(models_dir, "coef.csv"), clf.coef_, delimiter=",")
        np.savetxt(os.path.join(models_dir, "intercept.csv"), clf.intercept_, delimiter=",")
        pd.DataFrame([{"class": c} for c in clf.classes_]).to_csv(
            os.path.join(models_dir, "classes.csv"), index=False
        )

        print("✅ Exported SVM weights to CSV")

        # Debugging: run a test prediction at startup
        print("\n--- Startup Test Predictions ---")
        test_texts = [
            "Free markets and low taxes boost economic growth",
            "Workers deserve fair wages and strong unions"
        ]
        for t in test_texts:
            cleaned = clean_text(t)
            vec = vectorizer.transform([cleaned])
            pred = clf.predict(vec)[0]
            print(f"  '{t}' → {pred}")
        print("--- End Startup Tests ---\n")

    except FileNotFoundError as e:
        print(f"❌ Model files not found! Run 'python src/train_baseline.py' first.")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")


@app.get("/predict", response_model=PredictResponse)
async def predict_get(text: str):
    return await predict(PredictRequest(text=text))


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    print("🔥🔥🔥 PREDICT ENDPOINT HIT 🔥🔥🔥")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if clf is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python src/train_baseline.py' first.",
        )

    # 1. Check if the input is a YouTube URL
    youtube_url_pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?([a-zA-Z0-9_-]{11})"
    youtube_match = re.search(youtube_url_pattern, request.text)

    if youtube_match:
        video_id = youtube_match.group(1)
        print(f"📺 Detected YouTube Video ID: {video_id}. Fetching transcript...")
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = " ".join([t['text'] for t in transcript])
            cleaned_input = clean_text(full_text)
            print("✅ Transcript downloaded and cleaned successfully!")
        except Exception as e:
            print(f"❌ Could not download transcript: {e}")
            raise HTTPException(status_code=400, detail=f"Could not extract YouTube transcript: Make sure the video has closed captions enabled.")
    else:
        # Otherwise, process as standard text
        cleaned_input = clean_text(request.text)

    if not cleaned_input:
        raise HTTPException(
            status_code=400,
            detail="Text contained no valid words after cleaning."
        )

    # 2. Vectorize
    text_vec = vectorizer.transform([cleaned_input])

    # 3. Predict directly using sklearn
    try:
        prediction = clf.predict(text_vec)[0]

        logging.info(f"Prediction: {prediction}")
        print(f"Input: '{request.text[:80]}...' → Prediction: {prediction}")

        return PredictResponse(prediction=str(prediction))

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Error during model inference.")


# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)