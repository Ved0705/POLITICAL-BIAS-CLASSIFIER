import os
import re
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# Text cleaning function (SAME as data_prep.py and train_baseline.py)
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

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if clf is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python src/train_baseline.py' first.",
        )

    # 1. Clean input (SAME preprocessing as training)
    cleaned_input = clean_text(request.text)

    if not cleaned_input:
        raise HTTPException(status_code=400, detail="Text contained no valid words after cleaning.")

    # 2. Vectorize with the SAME TF-IDF vectorizer used during training
    text_vec = vectorizer.transform([cleaned_input])

    # 3. Predict
    try:
        prediction = clf.predict(text_vec)[0]
        print(f"Input: '{request.text[:80]}...' → Cleaned: '{cleaned_input[:80]}...' → Prediction: {prediction}")
        return PredictResponse(prediction=prediction)
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Error during model inference.")

# If executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
