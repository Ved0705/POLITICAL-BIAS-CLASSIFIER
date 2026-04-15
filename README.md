# 🏛️ Political Bias Classifier

The **Political Bias Classifier** is a full-stack Data Science application that employs Natural Language Processing (NLP) to detect political leanings in written and spoken text. By analyzing speeches, news articles, and social media posts, it accurately classifies content as **Left**, **Right**, or **Center**. This tool helps researchers, journalists, and readers identify subtle biases and understand how language shapes political perspectives.

## 🌟 Key Features
- **Accurate Bias Prediction**: Distinguishes text reliably into Left, Right, or Center categories.
- **Robust NLP Pipeline**: Features built-in data cleaning, preprocessing, and TF-IDF vectorization.
- **Support for Multimedia**: Contains utilities to process raw video transcripts and utilize the Gemini 2.0 Flash API for translations.
- **Interactive Web Interface**: A clean, accessible Single-Page Application (React) to instantly test out custom inputs.

## 🚀 Architecture Overview

The project is structured into 4 modular folders assuring scalability:

* **`classifier_model/` (Backend AI Service)**
  * Driven by **FastAPI** to serve predictions securely (`src/api.py`).
  * Employs `scikit-learn` (`LinearSVC`) for rapid classification, alongside an experimental Hugging Face Transformer logic (`distilbert`).
  * Features a fully containerized pipeline using **Docker**.
  * Contains reproducible scripts for acquiring datasets, cleaning text, and executing model training processes.
  * Integration with **R scripts** (`analysis.R`) allows for cross-stack matrix computing and prediction rendering.

* **`frontend/` (User Interface)**
  * Built with modern **React** + **Vite** for incredibly fast web delivery.
  * Connects to the robust backend API asynchronously to provide users with a seamless UI.

* **`transcript_translator/` & `transcript_generator/` (Language Utilities)**
  * Extra functionalities designed to extend the classifier's capabilities to international and cross-medium contexts using advanced LLMs.

## 🛠️ Getting Started

### 1. Running the FastAPI Backend
You can run the backend either via Docker or natively through Python.

**Option A - Native Python**
```bash
cd classifier_model
# Assuming a virtual environment is active:
pip install -r requirements.txt
python src/api.py
```
*(Runs naturally on port 8000)*

**Option B - Docker**
```bash
cd classifier_model
docker build -t pds-project .
docker run -p 8001:8001 pds-project
```
*(Runs on port 8001)*

### 2. Launching the React Frontend
Open another terminal process:
```bash
cd frontend
npm install
npm run dev
```

*Note: Ensure your `frontend/src/App.jsx` points to your backend's correct active port before making predictions!*

## 📚 Dataset Information
This project is built using the `cajcodes/political-bias` dataset curated from HuggingFace. Scripts like `download_data.py` and `data_prep.py` are included to re-pull, clean, and resolve existing label anomalies systematically.