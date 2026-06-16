# 🔍 ReviewLens: App Sentiment Classifier & NLP Insights Dashboard

ReviewLens is a production-grade NLP application designed to parse, classify, and extract actionable insights from app store reviews. Driven by a combination of traditional machine learning and state-of-the-art Deep Learning (RoBERTa Transformers), ReviewLens helps product and engineering teams cut through the noise, resolve software issues faster, and prevent user churn.

---

## 🎯 The Problem We Solve
**User reviews contain critical product signals, but they are buried in noise.**
- **The Scale Problem**: App stores receive thousands of reviews daily. Reading them manually is impossible.
- **Lost Signals**: Critical warnings about crash loops, billing issues, security flaws, and interface freezes are lost inside a sea of text.
- **Delayed Response**: Product teams often realize an update is broken only after their app store rating drops and churn spikes.

## 💡 The Solution & Benefits
ReviewLens automates customer feedback loops by turning raw text into structured product intelligence:
* ⏱️ **Reduces Time-To-Fix**: Instantly maps customer complaints to specific categories (e.g., billing, crashes, lag).
* 📱 **Version & Device Tracking**: Filters complaints by OS version and device models to pin down exactly what broke.
* 🧠 **Explainable AI (XAI)**: Visualizes word influence so you know exactly *why* a model classified a review a certain way.
* 🚀 **Boosts Team Velocity**: Product teams spend less time reading spreadsheets and more time fixing bugs.

---

## ✨ Features

### 🔍 1. Text Analyzer
Analyze individual or bulk reviews in real-time.
- **Single Review Analyzer**: Classifies text polarity (Positive, Negative, Neutral) with confidence scores, exact class probabilities, and word count.
- **Word Influence Analyzer (Explainable NLP)**:
  - *Traditional ML*: Highlights feature weights showing which words drove the classification.
  - *RoBERTa Transformer*: Displays gradient × embedding saliency maps to show which tokens the deep learning model focused on.
- **Keyword Issue Matcher**: Automatically scans text for issues and categorizes them into Bugs, Lags, Connectivity, Battery Drain, UI glitches, and more.
- **Batch Review Analyzer**: Paste lists of reviews and parse them in bulk with instant sentiment distribution graphs and styled tabular outputs.

### 📂 2. Interactive Data Explorer
A central workspace to manage and filter review databases.
- **Custom CSV Uploader**: Upload any review dataset, specify the text column, and automatically run multi-model sentiment predictions across the entire file.
- **Multi-Dimensional Filters**: Filter dataset by rating, sentiment, OS version, device model, specific issue tags, or search queries.
- **Report Downloader**: Download your filtered and classified reviews directly as a clean CSV report.

### 📊 3. Sentiment Analytics & Trends Dashboard
Deep dive into product health using rich data visualizations:
- **Net Sentiment Index (NSI)**: A core metric to track the overall satisfaction trajectory.
- **Sentiment Trend Line**: Line chart tracking positive, negative, and neutral sentiment ratios over time.
- **Complaints & Issues Breakdown**: Frequency metrics displaying the top categories of user frustrations.
- **Dynamic Word Cloud**: Interactive visualization of the most prominent words in user reviews.
- **Confusion Matrix Heatmap**: A cross-examination grid comparing user-given star ratings against model classifications to audit classifier quality.
- **Model Comparison metrics**: Real-time benchmarks for classification latency (ms) and dataset accuracy.

---

## 📂 Project Directory Structure

```
NLP_Final_Project/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependency list
├── README.md               # Setup and usage instructions (this file)
├── .gitignore              # Files ignored by git (e.g. models, datasets)
├── data/                   # Sample datasets
│   ├── googleplaystore.csv
│   └── googleplaystore_user_reviews.csv
├── models/                 # Serialized ML models and local Transformer configurations
│   ├── tfidf_model.joblib            # TF-IDF vectorizer
│   ├── logistic_regression_model.pkl # Logistic Regression model
│   ├── naive_bayes_model.pkl         # Multinomial Naive Bayes model
│   ├── cardiffnlp_roberta.pkl        # CardiffNLP RoBERTa serialized pkl
│   └── app_roberta_model/            # Local directory with Hugging Face RoBERTa files
├── notebooks/              # Jupyter notebook for model training & development
│   └── NLP_Project.ipynb
└── scripts/                # Utility scripts for model management and deployment
    ├── export_to_hf.py               # Exports local serialized models to standard HF format
    └── upload_model.py               # Uploads model folder to Hugging Face Hub
```

---

## 🤖 Model Performance Architecture

ReviewLens runs three classification engines to balance speed and accuracy:

| Classifier Model | Type | Real-Time Latency | Target Accuracy | Key Strengths |
| :--- | :--- | :--- | :--- | :--- |
| **RoBERTa (Transformer)** | Deep Learning | ~250ms | **93.5%** | Highest accuracy, captures context, negation, and slang. |
| **Logistic Regression** | Traditional ML | ~1.5ms | **87.1%** | Balanced speed/accuracy, highly interpretable word coefficients. |
| **Naive Bayes** | Traditional ML | ~0.8ms | **85.2%** | Ultra-lightweight, extremely fast training and execution. |

---

## 🚀 Getting Started

### 1. Setup and Installation
Clone the repository, navigate to the folder, and install dependencies from `requirements.txt`:

```bash
# Clone the repository
git clone https://github.com/Risikesan26/ReviewLens_NLP.git
cd ReviewLens_NLP

# Create a virtual environment (optional but recommended)
python -m venv venv
# Activate virtual environment:
# On Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# On macOS/Linux
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Running the Streamlit App
Launch the interactive dashboard locally:

```bash
streamlit run app.py
```
*Open the local URL generated in your terminal (usually `http://localhost:8501`) to start exploring.*

### 🛠️ Model Utilities
- **Export standard HF models**: Run `python scripts/export_to_hf.py` to extract weights from `models/cardiffnlp_roberta.pkl` and export them as standard configuration files in `models/my_hf_model/`.
- **Upload models to Hugging Face Hub**: Run `python scripts/upload_model.py` to push `models/app_roberta_model` to your Hugging Face space.
