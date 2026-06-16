# ReviewLens: App Sentiment Classifier

ReviewLens is a modern, premium Streamlit application designed for analyzing and visualizing sentiment in mobile app user reviews. The app implements multiple Machine Learning models and a Transformer model to classify reviews into **Positive**, **Neutral**, or **Negative** categories.

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

## 🚀 Getting Started

### 1. Prerequisites
- **Python**: Version `3.9` or `3.10` is recommended.
- **Git** & **Git LFS** (for downloading the RoBERTa model files if cloning the repo).

### 2. Setup and Installation
Clone the repository, navigate to the folder, and install dependencies from `requirements.txt`:

```bash
# Clone the repository
git clone <repository_url>
cd NLP_Final_Project

# Create a virtual environment (optional but recommended)
python -m venv venv
# Activate virtual environment:
# On Windows (Command Prompt)
venv\Scripts\activate
# On Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# On macOS/Linux
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Running the Streamlit App
To run the ReviewLens application:

```bash
streamlit run app.py
```

Open the local URL generated in your terminal (usually `http://localhost:8501`) to interact with the interface.

---

## 🤖 Models & Classification

ReviewLens supports three primary classifiers:
1. **Logistic Regression** (Traditional ML, fast, highly interpretable)
2. **Multinomial Naive Bayes** (Traditional ML, extremely lightweight)
3. **RoBERTa (Transformer)** (Deep Learning model optimized for social media/tweets, higher accuracy)

### Utility Scripts
Under `scripts/`, you'll find tools for maintaining the model deployment:
- **`scripts/export_to_hf.py`**: Extracts weights from `cardiffnlp_roberta.pkl` and formats them into standard Hugging Face configuration files in `models/my_hf_model/`.
- **`scripts/upload_model.py`**: Uploads standard Hugging Face model folder (`models/app_roberta_model`) to the Hugging Face Hub under your repository.
