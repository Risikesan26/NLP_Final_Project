import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, timedelta
from collections import Counter
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import joblib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReviewLens · App Sentiment Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  ML MODELS LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_ml_models():
    tfidf = joblib.load("models/tfidf_model.joblib")
    lr = joblib.load("models/logistic_regression_model.pkl")
    nb = joblib.load("models/naive_bayes_model.pkl")
    # Fix version compatibility for Logistic Regression
    if not hasattr(lr, "multi_class"):
        lr.multi_class = "auto"
    
    # Load RoBERTa (DistilBERT) model from the bundled local folder committed via Git LFS.
    # pytorch_model.bin is present so transformers loads it automatically on all versions.
    model_path = "models/app_roberta_model"
    roberta_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    roberta_tokenizer = AutoTokenizer.from_pretrained(model_path)
    roberta_model.eval()
    
    return tfidf, lr, nb, roberta_model, roberta_tokenizer

tfidf_vectorizer, lr_model, nb_model, roberta_model, roberta_tokenizer = load_ml_models()

# ── ML Helper Functions ───────────────────────────────────────────────────────
def predict_ml(model, text: str) -> dict:
    if not text.strip():
        return {
            "sentiment": "Neutral",
            "emoji": "😐",
            "confidence": 0,
            "probabilities": {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0},
            "word_count": 0
        }
    
    cleaned = clean_text(text)
    words = cleaned.split()
    vectorized = tfidf_vectorizer.transform([text])
    pred = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    classes = list(model.classes_)
    pred_idx = classes.index(pred)
    confidence = int(proba[pred_idx] * 100)
    
    emojis = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}
    return {
        "sentiment": pred,
        "emoji": emojis.get(pred, "😐"),
        "confidence": confidence,
        "probabilities": {classes[i]: float(proba[i]) for i in range(len(classes))},
        "word_count": len(words)
    }

def predict_roberta(text: str) -> dict:
    if not text.strip():
        return {
            "sentiment": "Neutral",
            "emoji": "😐",
            "confidence": 0,
            "probabilities": {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0},
            "word_count": 0
        }

    inputs = roberta_tokenizer(text, return_tensors="pt")

    # Pop token_type_ids if present (needed for DistilBERT models)
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        outputs = roberta_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze().tolist()
        
    # Dynamically extract classes from model configuration
    if hasattr(roberta_model.config, "id2label") and roberta_model.config.id2label:
        id2label = roberta_model.config.id2label
        classes = [id2label[i] for i in sorted(id2label.keys(), key=lambda x: int(x))]
    else:
        classes = ["Negative", "Neutral", "Positive"]
        
    pred_idx = int(np.argmax(probs))
    pred = classes[pred_idx]
    confidence = int(probs[pred_idx] * 100)
    
    emojis = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}
    return {
        "sentiment": pred,
        "emoji": emojis.get(pred, "😐"),
        "confidence": confidence,
        "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
        "word_count": len(text.split())
    }


# ── Word Influence Helpers ────────────────────────────────────────────────────
def get_word_influence_ml(model, text: str, predicted_class: str):
    """Return (positive_words, negative_words) lists of (word, score) tuples
    by multiplying TF-IDF weights with the model's per-class coefficients."""
    try:
        vectorized = tfidf_vectorizer.transform([text])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        classes = list(model.classes_)
        if predicted_class not in classes:
            return [], []
        class_idx = classes.index(predicted_class)

        # LR has coef_; NB has feature_log_prob_
        if hasattr(model, "coef_"):
            coefs = model.coef_[class_idx]
        else:
            coefs = model.feature_log_prob_[class_idx]

        nonzero_indices = vectorized.nonzero()[1]
        word_scores = []
        for idx in nonzero_indices:
            word = feature_names[idx]
            tfidf_val = float(vectorized[0, idx])
            influence = tfidf_val * float(coefs[idx])
            word_scores.append((word, influence))

        word_scores.sort(key=lambda x: x[1], reverse=True)
        positive = [(w, s) for w, s in word_scores if s > 0][:6]
        negative = [(w, abs(s)) for w, s in word_scores if s < 0][:6]
        negative.sort(key=lambda x: x[1], reverse=True)
        return positive, negative
    except Exception:
        return [], []


def get_word_influence_roberta(text: str, predicted_class: str):
    """Gradient × embedding saliency for RoBERTa/DistilBERT.
    Returns (top_tokens, []) where top_tokens are (token, score) sorted by
    absolute influence on the predicted class logit."""
    try:
        inputs = roberta_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )
        inputs.pop("token_type_ids", None)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        embed_layer = roberta_model.get_input_embeddings()

        with torch.enable_grad():
            roberta_model.zero_grad()
            embeddings = embed_layer(input_ids).detach().requires_grad_(True)
            outputs = roberta_model(inputs_embeds=embeddings, attention_mask=attention_mask)
            logits = outputs.logits

            if hasattr(roberta_model.config, "id2label") and roberta_model.config.id2label:
                id2label = roberta_model.config.id2label
                classes = [id2label[i] for i in sorted(id2label.keys(), key=lambda x: int(x))]
            else:
                classes = ["Negative", "Neutral", "Positive"]

            if predicted_class not in classes:
                return [], []
            class_idx = classes.index(predicted_class)

            logits[0, class_idx].backward()
            # Saliency = |grad · embedding| summed over hidden dim
            saliency = (embeddings.grad * embeddings).sum(dim=-1).abs().squeeze()

        tokens = roberta_tokenizer.convert_ids_to_tokens(input_ids[0])
        special = {roberta_tokenizer.cls_token, roberta_tokenizer.sep_token,
                   roberta_tokenizer.pad_token, "[CLS]", "[SEP]", "[PAD]",
                   "<s>", "</s>", "<pad>"}
        token_scores = []
        for tok, score in zip(tokens, saliency.tolist()):
            if tok in special:
                continue
            clean = tok.replace("##", "").replace("Ġ", "").replace("▁", "").strip()
            if clean:
                token_scores.append((clean, score))

        token_scores.sort(key=lambda x: x[1], reverse=True)
        return token_scores[:8], []
    except Exception:
        return [], []


# ── Global CSS — STARLIGHT PREMIUM THEME ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem 2rem; max-width: 1280px; }

/* ── Color tokens — Starlight Premium ── */
:root {
    --bg:        #0B1120;
    --surface:   #141C2E;
    --surface2:  #1E293B;
    --border:    #2A3A50;
    --accent:    #FBBF24;
    --accent2:   #38BDF8;
    --pos:       #34D399;
    --neg:       #FB7185;
    --neu:       #A78BFA;
    --text:      #F8FAFC;
    --muted:     #94A3B8;
    --glow-gold: rgba(251, 191, 36, 0.15);
    --glow-blue: rgba(56, 189, 248, 0.15);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #141C2E 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebarNav"] { display: none; }

/* ── Main bg ── */
.stApp { background: var(--bg); }

/* ── Typography helpers ── */
.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 800;
    line-height: 1.1;
    color: var(--text);
    letter-spacing: -1.5px;
}
.hero-title span { 
    color: transparent;
    background: linear-gradient(135deg, var(--accent), #F59E0B);
    -webkit-background-clip: text;
    background-clip: text;
}

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
    display: inline-block;
    background: var(--glow-gold);
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid rgba(251, 191, 36, 0.3);
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,.2);
    backdrop-filter: blur(10px);
}

.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform .2s, border-color .2s, box-shadow .2s;
    box-shadow: 0 4px 20px rgba(0,0,0,.2);
}
.stat-card:hover { 
    transform: translateY(-3px); 
    border-color: var(--accent);
    box-shadow: 0 8px 30px var(--glow-gold);
}
.stat-number {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
}
.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 6px;
    letter-spacing: .5px;
    text-transform: uppercase;
}

.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: .5px;
    border: 1px solid;
}
.badge-pos { background: rgba(52,211,153,.1); color: var(--pos); border-color: rgba(52,211,153,.3); }
.badge-neg { background: rgba(251,113,133,.1); color: var(--neg); border-color: rgba(251,113,133,.3); }
.badge-neu { background: rgba(167,139,250,.1); color: var(--neu); border-color: rgba(167,139,250,.3); }

.result-hero {
    background: linear-gradient(135deg, var(--surface), var(--surface2));
    border: 1px solid var(--accent);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 40px var(--glow-gold);
}
.result-hero::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: var(--glow-gold);
    filter: blur(40px);
    z-index: 0;
}

.pill-tag {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 4px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text);
    margin: 3px;
}
.pill-tag.hot { border-color: var(--neg); color: var(--neg); background: rgba(251,113,133,.1); }
.pill-tag.good { border-color: var(--pos); color: var(--pos); background: rgba(52,211,153,.1); }

.timeline-bar {
    height: 8px;
    border-radius: 4px;
    background: var(--surface2);
    position: relative;
    overflow: hidden;
}
.timeline-fill {
    height: 100%;
    border-radius: 4px;
    transition: width .6s ease;
}

/* ── Streamlit widget overrides ── */
.stTextArea textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--glow-gold) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #FBBF24, #F59E0B) !important;
    color: #0B1120 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-size: 0.95rem !important;
    letter-spacing: .5px !important;
    transition: transform .15s, box-shadow .15s !important;
    box-shadow: 0 4px 15px var(--glow-gold) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(251, 191, 36, 0.3) !important;
}

[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    box-shadow: 0 4px 15px rgba(0,0,0,.15) !important;
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.8rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 800 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    border-radius: 12px 12px 0 0 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.8rem 1.4rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 3px solid var(--accent) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 1.5rem !important;
}

/* ── Selectbox / multiselect ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ── Progress bars ── */
.stProgress > div > div { background: var(--surface2) !important; }
.stProgress > div > div > div { background: linear-gradient(90deg, var(--accent), #F59E0B) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SENTIMENT ENGINE
# ══════════════════════════════════════════════════════════════════════════════
ISSUE_CATEGORIES = {
    "🐛 Bugs & Errors":       ["bug", "bugs", "error", "errors", "glitch", "glitchy", "broken", "not working", "doesn't work"],
    "💥 Crashes & Freezes":   ["crash", "crashes", "freeze", "frozen", "stuck", "hang", "hangs"],
    "🐢 Performance":         ["slow", "lag", "laggy", "loading", "speed", "fast", "performance"],
    "🔋 Battery & Heat":      ["battery", "drain", "overheat", "overheating", "hot", "heat"],
    "🎨 UI / Design":         ["ui", "interface", "design", "layout", "ugly", "beautiful", "look", "theme", "dark mode"],
    "🔒 Privacy & Security":  ["privacy", "data", "permission", "secure", "security", "tracking"],
    "💳 Payments":            ["pay", "payment", "subscription", "purchase", "refund", "charge", "expensive", "price", "overpriced"],
    "📶 Connectivity":        ["wifi", "network", "internet", "connection", "offline", "sync"],
    "🔔 Notifications":       ["notification", "notifications", "alert", "spam", "push"],
    "⚙️ Features":            ["feature", "features", "option", "update", "missing", "add", "remove"],
}


def clean_text(text: str) -> str:
    return re.sub(r"[^\w\s']", " ", text.lower())


def analyze_sentiment(text: str, model_name: str = "Logistic Regression") -> dict:
    cleaned = clean_text(text)
    words = cleaned.split()
    text_lower = text.lower()

    # 1. Detect issue categories (keyword matching)
    detected_issues = {}
    for cat, keywords in ISSUE_CATEGORIES.items():
        hits = [k for k in keywords if k in text_lower]
        if hits:
            detected_issues[cat] = hits

    # 2. Predict sentiment based on the chosen model
    if model_name == "Logistic Regression":
        res = predict_ml(lr_model, text)
    elif model_name == "Multinomial Naive Bayes":
        res = predict_ml(nb_model, text)
    elif model_name == "RoBERTa (Transformer)":
        res = predict_roberta(text)
    else:
        res = predict_ml(lr_model, text)

    return {
        "sentiment": res["sentiment"],
        "emoji": res["emoji"],
        "pos_score": round(res["probabilities"].get("Positive", 0.0) * 10, 1),
        "neg_score": round(res["probabilities"].get("Negative", 0.0) * 10, 1),
        "confidence": res["confidence"],
        "detected_issues": detected_issues,
        "word_count": len(words),
        "probabilities": res["probabilities"]
    }


def classify_batch(reviews: list[str]) -> pd.DataFrame:
    model_name = st.session_state.get("classifier_model", "Logistic Regression")
    records = []
    for i, r in enumerate(reviews):
        r = r.strip()
        if not r:
            continue
        res = analyze_sentiment(r, model_name)
        records.append({
            "Review": r[:120] + ("…" if len(r) > 120 else ""),
            "Sentiment": res["sentiment"],
            "Confidence": res["confidence"],
            "Pos Score": res["pos_score"],
            "Neg Score": res["neg_score"],
            "Issues": ", ".join(res["detected_issues"].keys()) if res["detected_issues"] else "None",
        })
    return pd.DataFrame(records)



# ══════════════════════════════════════════════════════════════════════════════
#  DEMO DATA
# ══════════════════════════════════════════════════════════════════════════════
DEMO_REVIEWS = [
    "This app is absolutely amazing! The UI is clean and everything loads fast. Love the dark mode.",
    "Keeps crashing every time I open it. Terrible experience. Uninstalling now.",
    "Pretty average app. Does what it says but nothing special. Decent performance.",
    "Battery drain is insane! My phone gets hot within 10 minutes of using this app.",
    "Best productivity app I've ever used. Smooth, intuitive, and well designed.",
    "Laggy interface, tons of bugs. The payment system doesn't work either. Scam!",
    "Works fine for basic needs. Notifications are a bit spammy though.",
    "Outstanding update! All previous issues fixed. Now runs perfectly on my device.",
    "Privacy is a huge concern. Too many permissions required. Not happy about data collection.",
    "Great app overall. Minor UI glitches but nothing that ruins the experience.",
    "Frozen twice today. Can't even load the main screen. Please fix ASAP.",
    "Love the new features in this update. The sync across devices is brilliant!",
]


def generate_mock_reviews(n=100):
    np.random.seed(42)
    random.seed(42)
    
    devices = ["iPhone 15 Pro", "Samsung S24 Ultra", "Google Pixel 8 Pro", "OnePlus 12", "iPad Pro", "Samsung Galaxy Tab S9"]
    versions = ["2.4.1", "2.4.0", "2.3.5", "2.2.0", "2.1.1", "2.0.0"]
    names = ["Alex M.", "Sarah K.", "David L.", "Emily R.", "James P.", "Jessica T.", "Michael B.", "Sophia W.", "Daniel H.", "Olivia C.", "Chris T.", "Anna S."]
    
    positive_templates = [
        "This app is absolutely amazing! The UI is clean and everything loads fast. Love the dark mode.",
        "Best productivity app I've ever used. Smooth, intuitive, and well designed.",
        "Outstanding update! All previous issues fixed. Now runs perfectly on my device.",
        "Love the new features in this update. The sync across devices is brilliant!",
        "Great app overall. Minor UI glitches but nothing that ruins the experience.",
        "Very useful app, it has completely changed my daily workflow. Highly recommended!",
        "Clean design, simple to navigate, and does exactly what it promises. Five stars!",
        "I enjoy using this application daily. The widgets are beautiful and extremely responsive.",
        "Super fast performance and excellent design. The support team is also very helpful.",
        "Stable, reliable, and keeps getting better with every update. Well done to the team!"
    ]
    
    negative_templates = [
        "Keeps crashing every time I open it. Terrible experience. Uninstalling now.",
        "Battery drain is insane! My phone gets hot within 10 minutes of using this app.",
        "Laggy interface, tons of bugs. The payment system doesn't work either. Scam!",
        "Frozen twice today. Can't even load the main screen. Please fix ASAP.",
        "Privacy is a huge concern. Too many permissions required. Not happy about data collection.",
        "Highly disappointed. The new update broke the widgets and they won't sync.",
        "Expensive subscription and not worth the money. Overpriced and useless.",
        "Tons of errors when uploading data. Connection timed out. Avoid this app.",
        "Annoying notifications and push alerts that cannot be disabled. Extremely spammy.",
        "Very slow load times. The UI looks outdated and is hard to navigate."
    ]
    
    neutral_templates = [
        "Pretty average app. Does what it says but nothing special. Decent performance.",
        "Works fine for basic needs. Notifications are a bit spammy though.",
        "It's okay but could be much better. Some features are missing in this version.",
        "Decent app, but the price is a bit high. Hope they add more templates.",
        "The interface is fine, but the battery usage is slightly higher than expected.",
        "Not bad, but not outstanding either. Just average features.",
        "An okay tool for tasks. It works, but it lags occasionally.",
        "Needs more customization options. Currently it is a bit too simple.",
        "The sync works sometimes but not always. Average experience.",
        "I like the design but it has some minor bugs. Hope they fix it soon."
    ]
    
    records = []
    start_date = datetime.now() - timedelta(days=30)
    
    for i in range(n):
        rand_val = random.random()
        if rand_val < 0.45:  # 45% positive
            text = random.choice(positive_templates)
            rating = random.choice([4, 5])
        elif rand_val < 0.80:  # 35% negative
            text = random.choice(negative_templates)
            rating = random.choice([1, 2])
        else:  # 20% neutral
            text = random.choice(neutral_templates)
            rating = 3
            
        device = random.choice(devices)
        version = random.choice(versions)
        name = random.choice(names)
        
        days_offset = random.randint(0, 30)
        hours_offset = random.randint(0, 23)
        minutes_offset = random.randint(0, 59)
        date = start_date + timedelta(days=days_offset, hours=hours_offset, minutes=minutes_offset)
        
        res_lr = analyze_sentiment(text, "Logistic Regression")
        res_nb = analyze_sentiment(text, "Multinomial Naive Bayes")
        res_rob = analyze_sentiment(text, "RoBERTa (Transformer)")
        
        records.append({
            "Date": date,
            "Username": name,
            "Rating": rating,
            "Device": device,
            "Version": version,
            "Review": text,
            
            "LR Sentiment": res_lr["sentiment"],
            "LR Confidence": res_lr["confidence"],
            "LR Pos Score": res_lr["pos_score"],
            "LR Neg Score": res_lr["neg_score"],
            
            "NB Sentiment": res_nb["sentiment"],
            "NB Confidence": res_nb["confidence"],
            "NB Pos Score": res_nb["pos_score"],
            "NB Neg Score": res_nb["neg_score"],
            
            "RoBERTa Sentiment": res_rob["sentiment"],
            "RoBERTa Confidence": res_rob["confidence"],
            "RoBERTa Pos Score": res_rob["pos_score"],
            "RoBERTa Neg Score": res_rob["neg_score"],
            
            "Issues": ", ".join(res_lr["detected_issues"].keys()) if res_lr["detected_issues"] else "None",
        })
        
    df = pd.DataFrame(records)
    df = df.sort_values(by="Date", ascending=False).reset_index(drop=True)
    return df


def get_trend_data_from_df(df):
    if df.empty:
        return pd.DataFrame(columns=["Date", "Positive", "Negative", "Neutral", "Total"])
    
    df_copy = df.copy()
    df_copy["DateOnly"] = df_copy["Date"].dt.date
    
    pivot = df_copy.groupby(["DateOnly", "Sentiment"]).size().unstack(fill_value=0)
    
    for col in ["Positive", "Negative", "Neutral"]:
        if col not in pivot.columns:
            pivot[col] = 0
            
    pivot["Total"] = pivot["Positive"] + pivot["Negative"] + pivot["Neutral"]
    
    pivot_pct = pivot.copy()
    for col in ["Positive", "Negative", "Neutral"]:
        pivot_pct[col] = (pivot[col] / (pivot["Total"] + 1e-5) * 100).round(1)
        
    pivot_pct = pivot_pct.reset_index().rename(columns={"DateOnly": "Date"})
    pivot_pct["Total"] = pivot["Total"].values
    pivot_pct["Date"] = pd.to_datetime(pivot_pct["Date"])
    pivot_pct = pivot_pct.sort_values(by="Date").reset_index(drop=True)
    return pivot_pct


# Initialize session state
if "mock_data" not in st.session_state:
    st.session_state["mock_data"] = generate_mock_reviews(100)

if "custom_data" not in st.session_state:
    st.session_state["custom_data"] = None

if "dataset_selection" not in st.session_state:
    st.session_state["dataset_selection"] = "Demo Dataset"

# Select active dataset
if st.session_state["custom_data"] is not None and st.session_state["dataset_selection"] == "Uploaded Dataset":
    active_df = st.session_state["custom_data"]
else:
    active_df = st.session_state["mock_data"]


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 .5rem 0'>
        <div style='font-family:Plus Jakarta Sans,sans-serif;font-size:1.35rem;font-weight:800;color:#FBBF24;letter-spacing:-0.5px'>
            🔍 ReviewLens
        </div>
        <div style='font-size:0.75rem;color:#94A3B8;margin-top:2px'>App Sentiment Classifier</div>
    </div>
    <hr style='border-color:#2A3A50;margin:0.8rem 0'>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Navigate",
        ["🏠 Home", "🔍 Text Analyzer", "📂 Data Explorer", "📊 Visualizations", "ℹ️ Model Info"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#2A3A50;margin:0.8rem 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.75rem;color:#94A3B8;margin-bottom:0.4rem'>Classifier Model</div>", unsafe_allow_html=True)
    classifier_model = st.selectbox(
        "Classifier Model",
        ["Logistic Regression", "Multinomial Naive Bayes", "RoBERTa (Transformer)"],
        label_visibility="collapsed",
        key="classifier_model"
    )


    st.markdown("<hr style='border-color:#2A3A50;margin:0.8rem 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.75rem;color:#94A3B8;margin-bottom:0.4rem'>Active Dataset</div>", unsafe_allow_html=True)
    
    dataset_options = ["Demo Dataset"]
    if st.session_state["custom_data"] is not None:
        dataset_options.append("Uploaded Dataset")
        
    selected_ds = st.radio(
        "Active Dataset",
        dataset_options,
        label_visibility="collapsed"
    )
    st.session_state["dataset_selection"] = selected_ds
    
    if selected_ds == "Uploaded Dataset" and st.session_state["custom_data"] is not None:
        active_df = st.session_state["custom_data"]
        st.info("Using uploaded custom reviews.")
    else:
        active_df = st.session_state["mock_data"]
        st.info("Using demo reviews (100 items).")
        
    # Dynamic mapping of active predictions based on selected classifier model
    if not active_df.empty:
        active_df = active_df.copy()
        prefix = {
            "Logistic Regression": "LR",
            "Multinomial Naive Bayes": "NB",
            "RoBERTa (Transformer)": "RoBERTa"
        }.get(classifier_model, "LR")
        
        active_df["Sentiment"] = active_df[f"{prefix} Sentiment"]
        active_df["Confidence"] = active_df[f"{prefix} Confidence"]
        active_df["Pos Score"] = active_df[f"{prefix} Pos Score"]
        active_df["Neg Score"] = active_df[f"{prefix} Neg Score"]

# ── Real-Time Metrics & Latency Calculations ─────────────────────────────────
import time

# Initialize latencies dict in session state
if "latencies" not in st.session_state:
    st.session_state["latencies"] = {}

# Make sure we measure the latency for all three models in real-time
for m in ["Multinomial Naive Bayes", "Logistic Regression", "RoBERTa (Transformer)"]:
    if m not in st.session_state["latencies"]:
        dummy_text = "This is a dummy test review to measure latency in real-time."
        # Warmup
        _ = analyze_sentiment(dummy_text, m)
        t0 = time.perf_counter()
        for _ in range(3):
            _ = analyze_sentiment(dummy_text, m)
        t1 = time.perf_counter()
        st.session_state["latencies"][m] = ((t1 - t0) / 3) * 1000

# Compute real-time dataset accuracy for each model
realtime_accuracies = {}
if not active_df.empty:
    y_true = active_df["Rating"].apply(lambda r: "Positive" if r >= 4 else ("Negative" if r <= 2 else "Neutral"))
    for model_name, prefix in [
        ("Logistic Regression", "LR"),
        ("Multinomial Naive Bayes", "NB"),
        ("RoBERTa (Transformer)", "RoBERTa")
    ]:
        col_pred = f"{prefix} Sentiment"
        if col_pred in active_df.columns:
            acc = (y_true == active_df[col_pred]).mean() * 100
            realtime_accuracies[model_name] = acc
        else:
            realtime_accuracies[model_name] = 85.0
else:
    realtime_accuracies = {
        "Logistic Regression": 87.1,
        "Multinomial Naive Bayes": 85.2,
        "RoBERTa (Transformer)": 93.5
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div style='padding:3rem 0 1.5rem 0'>
        <div class='section-label'>NLP Classifier Dashboard</div>
        <div class='hero-title'>ReviewLens<br><span>Sentiment Classifier</span></div>
        <div style='font-size:1.05rem;color:#94a3b8;margin-top:1rem;max-width:640px;line-height:1.7'>
            ReviewLens is an intelligent machine learning and deep learning natural language processing application designed to analyze mobile app reviews from Google Play & the App Store. The system instantly classifies sentiment, detects common product issues, and surfaces trends so product teams can act before users churn.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='stat-card'>
            <div class='stat-number' style='color:#FBBF24'>{len(active_df)}</div>
            <div class='stat-label'>Analyzed Reviews</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='stat-card'>
            <div class='stat-number' style='color:#38BDF8'>{len(ISSUE_CATEGORIES)}</div>
            <div class='stat-label'>Issue Categories</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        pos_ratio = (active_df["Sentiment"] == "Positive").sum() / len(active_df) if len(active_df) > 0 else 0
        st.markdown(f"""<div class='stat-card'>
            <div class='stat-number' style='color:#34D399'>{pos_ratio*100:.1f}%</div>
            <div class='stat-label'>Positive Ratio</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='stat-card'>
            <div class='stat-number' style='color:#A78BFA'>{st.session_state["latencies"].get(classifier_model, 2.0):.2f} ms</div>
            <div class='stat-label'>Classification Speed</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Problem + How to use
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("""
        <div class='card' style='height: 100%; min-height: 380px;'>
            <div class='section-label'>What Problem We're Solving</div>
            <div style='font-family:Plus Jakarta Sans,sans-serif;font-size:1.2rem;font-weight:700;color:#F8FAFC;margin-bottom:.8rem'>
                Reviews are gold — but often go unread
            </div>
            <div style='color:#94a3b8;font-size:0.9rem;line-height:1.8'>
                The average app on Google Play & App Store receives <b>thousands</b> of reviews per month. 
                Product teams manually read a tiny fraction, missing critical signals about crashes, 
                billing issues, and UX pain-points — <em>until users leave.</em>
            </div>
            <br>
            <div style='color:#94a3b8;font-size:0.9rem;line-height:1.8'>
                ReviewLens automates this signal extraction by utilizing advanced Machine Learning and Transformer models. 
                It surfaces immediate product bugs, security vulnerabilities, connectivity drops, and interface lags 
                so your team can focus on fixes, not reading spreadsheets.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class='card' style='height: 100%; min-height: 380px;'>
            <div class='section-label'>How to Use the App</div>
            <div style='font-family:Plus Jakarta Sans,sans-serif;font-size:1.2rem;font-weight:700;color:#F8FAFC;margin-bottom:.8rem'>
                Quick Start Steps
            </div>
            <div style='color:#94a3b8;font-size:0.88rem;line-height:1.8'>
                <div style='display:flex;align-items:flex-start;gap:.7rem;margin-bottom:.7rem'>
                    <span style='background:#1E293B;border:1px solid #2A3A50;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-family:Plus Jakarta Sans,sans-serif;font-weight:700;font-size:.7rem;color:#FBBF24;flex-shrink:0'>1</span>
                    <span><b>Select a Page:</b> Use the sidebar selectbox to navigate between Home, Text Analyzer, Data Explorer, Visualizations, and Model Info.</span>
                </div>
                <div style='display:flex;align-items:flex-start;gap:.7rem;margin-bottom:.7rem'>
                    <span style='background:#1E293B;border:1px solid #2A3A50;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-family:Plus Jakarta Sans,sans-serif;font-weight:700;font-size:.7rem;color:#38BDF8;flex-shrink:0'>2</span>
                    <span><b>Analyze Text:</b> Input a single review or multiple lines of text in the <b>Text Analyzer</b> tab to get instant sentiment tags and keyword matching highlights.</span>
                </div>
                <div style='display:flex;align-items:flex-start;gap:.7rem;margin-bottom:.7rem'>
                    <span style='background:#1E293B;border:1px solid #2A3A50;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-family:Plus Jakarta Sans,sans-serif;font-weight:700;font-size:.7rem;color:#34D399;flex-shrink:0'>3</span>
                    <span><b>Manage Datasets:</b> Toggle between the built-in <b>Demo Dataset</b> or upload your own CSV files in the <b>Data Explorer</b> page.</span>
                </div>
                <div style='display:flex;align-items:flex-start;gap:.7rem'>
                    <span style='background:#1E293B;border:1px solid #2A3A50;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-family:Plus Jakarta Sans,sans-serif;font-weight:700;font-size:.7rem;color:#A78BFA;flex-shrink:0'>4</span>
                    <span><b>Filter & Export:</b> Apply date filters, search terms, and rating criteria to export classified reports or view dynamic trends on the <b>Visualizations</b> dashboard.</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Team
    st.markdown("<br><div class='section-label'>Team Members</div>", unsafe_allow_html=True)
    team = [
        ("Kirtanan", "ML Engineer", "Sentiment model & NLP pipeline development"),
        ("Risikesan", "UI/UX Designer", "Interface design, custom CSS styling, & data visualization"),
        ("Shamalan", "Data Scientist", "Feature engineering, dataset cleaning, & trend analysis"),
        ("Nafiz Shukri", "Role", "Description of contribution and responsibilities"),
    ]
    t1, t2, t3, t4 = st.columns(4)
    for col, (name, role, desc) in zip([t1, t2, t3, t4], team):
        col.markdown(f"""
        <div class='card' style='text-align:center'>
            <div style='font-family:Plus Jakarta Sans,sans-serif;font-weight:700;color:#F8FAFC;margin-top:.4rem'>{name}</div>
            <div style='color:#FBBF24;font-size:.75rem;margin:.2rem 0'>{role}</div>
            <div style='color:#94A3B8;font-size:.78rem'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — TEXT ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Text Analyzer":
    st.markdown("""
    <div style='padding:2rem 0 1rem 0'>
        <div class='section-label'>Review Analysis Engine</div>
        <div style='font-family:Plus Jakarta Sans,sans-serif;font-size:1.8rem;font-weight:800;color:#F8FAFC'>
            Text Analyzer
        </div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.tabs(["🔍 Single Review Analyzer", "📋 Batch Review Analyzer"])

    with mode[0]:
        review_text = st.text_area(
            "Review Text Input",
            value="",
            height=140,
            placeholder="Paste a review here to analyze its sentiment & extract issue tags…",
            label_visibility="collapsed",
            key="single_analyzer_text"
        )

        analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=False, key="single_analyze_btn")

        if (analyze_btn or review_text.strip()) and review_text.strip():
            classifier_model = st.session_state.get("classifier_model", "Logistic Regression")
            result = analyze_sentiment(review_text, classifier_model)
            sent = result["sentiment"]

            color_map = {"Positive": "#34D399", "Negative": "#FB7185", "Neutral": "#A78BFA"}
            badge_map = {"Positive": "badge-pos", "Negative": "badge-neg", "Neutral": "badge-neu"}
            emoji = result["emoji"]
            color = color_map[sent]
            badge_cls = badge_map[sent]
            conf = result["confidence"]

            # Result hero card
            st.markdown(f"""
            <div class='result-hero'>
                <div style='display:flex;align-items:center;gap:1rem;margin-bottom:1.2rem'>
                    <div style='font-size:3rem'>{emoji}</div>
                    <div>
                        <span class='badge {badge_cls}' style='font-size:.85rem;padding:4px 16px'>{sent}</span>
                        <div style='font-family:Plus Jakarta Sans,sans-serif;font-size:1.6rem;font-weight:800;color:{color};margin-top:.3rem'>
                            {conf}% Confident
                        </div>
                    </div>
                </div>
                <div style='background:rgba(255,255,255,.03);border-radius:8px;padding:.8rem 1rem;font-size:.88rem;color:#94a3b8;font-style:italic;line-height:1.6'>
                    "{review_text}"
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Scores
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Positive Score", result["pos_score"])
            mc2.metric("Negative Score", result["neg_score"])
            mc3.metric("Word Count", result["word_count"])

            st.markdown("<br>", unsafe_allow_html=True)

            # Two-column detail
            left, right = st.columns(2)

            with left:
                st.markdown("<div class='section-label'>Class Probabilities</div>", unsafe_allow_html=True)
                if result.get("probabilities") is not None:
                    for cls_name, prob in result["probabilities"].items():
                        prob_pct = int(prob * 100)
                        prob_color = color_map.get(cls_name, "#F8FAFC")
                        st.markdown(f"""
                        <div style='margin-bottom:.6rem'>
                            <div style='display:flex;justify-content:space-between;margin-bottom:2px'>
                                <span style='color:#F8FAFC;font-size:.82rem'>{cls_name}</span>
                                <span style='color:#94a3b8;font-size:.78rem'>{prob_pct}%</span>
                            </div>
                            <div class='timeline-bar'>
                                <div class='timeline-fill' style='width:{prob_pct}%;background:{prob_color}'></div>
                            </div>
                        </div>""", unsafe_allow_html=True)

            with right:
                st.markdown("<div class='section-label'>Detected Issue Categories</div>", unsafe_allow_html=True)
                if result["detected_issues"]:
                    for cat, keywords in result["detected_issues"].items():
                        kw_str = ", ".join(keywords[:3])
                        st.markdown(f"""
                        <div style='display:flex;justify-content:space-between;align-items:center;
                             padding:.5rem .7rem;background:#1E293B;border-radius:8px;margin-bottom:.4rem'>
                            <span style='color:#F8FAFC;font-size:.85rem'>{cat}</span>
                            <span style='color:#94A3B8;font-size:.75rem'>{kw_str}</span>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:#94A3B8;font-size:.85rem'>No specific product issues detected in text.</span>", unsafe_allow_html=True)

            # ── Word Influence Analysis ──────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>🔑 Word Influence Analysis</div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:0.8rem;color:#94a3b8;margin-bottom:0.8rem'>"
                "Which specific words in your review drove the sentiment prediction."
                "</div>",
                unsafe_allow_html=True
            )

            if classifier_model == "RoBERTa (Transformer)":
                top_tokens, _ = get_word_influence_roberta(review_text, sent)
                if top_tokens:
                    max_score = max(s for _, s in top_tokens) or 1
                    html_inf = "<div style='background:#141C2E;border:1px solid #2A3A50;border-radius:12px;padding:1rem'>"
                    html_inf += "<div style='font-size:0.78rem;color:#94a3b8;margin-bottom:0.6rem'>Most influential tokens (gradient × embedding saliency):</div>"
                    for tok, score in top_tokens:
                        bar_w = int(score / max_score * 100)
                        html_inf += f"""
                        <div style='margin-bottom:0.45rem'>
                            <div style='display:flex;justify-content:space-between;margin-bottom:2px'>
                                <span style='color:#F8FAFC;font-family:JetBrains Mono,monospace;font-size:.82rem'>{tok}</span>
                                <span style='color:#94a3b8;font-size:.75rem'>{score:.3f}</span>
                            </div>
                            <div class='timeline-bar'>
                                <div class='timeline-fill' style='width:{bar_w}%;background:#38BDF8'></div>
                            </div>
                        </div>"""
                    html_inf += "</div>"
                    st.markdown(html_inf, unsafe_allow_html=True)
                else:
                    st.info("Word influence unavailable for this input.")
            else:
                # LR or NB
                ml_model = lr_model if classifier_model == "Logistic Regression" else nb_model
                pos_words, neg_words = get_word_influence_ml(ml_model, review_text, sent)
                wi_col_l, wi_col_r = st.columns(2)
                with wi_col_l:
                    st.markdown(
                        f"<div style='font-size:0.82rem;font-weight:700;color:#34D399;margin-bottom:0.5rem'>"
                        f"🟢 Supporting '{sent}'</div>",
                        unsafe_allow_html=True
                    )
                    if pos_words:
                        max_s = max(s for _, s in pos_words) or 1
                        for word, score in pos_words:
                            bar_w = int(score / max_s * 100)
                            st.markdown(f"""
                            <div style='margin-bottom:0.4rem'>
                                <div style='display:flex;justify-content:space-between;margin-bottom:2px'>
                                    <span style='color:#F8FAFC;font-family:JetBrains Mono,monospace;font-size:.8rem'>{word}</span>
                                    <span style='color:#94a3b8;font-size:.72rem'>{score:.3f}</span>
                                </div>
                                <div class='timeline-bar'>
                                    <div class='timeline-fill' style='width:{bar_w}%;background:#34D399'></div>
                                </div>
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color:#94A3B8;font-size:.82rem'>No strong positive signals found.</span>", unsafe_allow_html=True)
                with wi_col_r:
                    st.markdown(
                        "<div style='font-size:0.82rem;font-weight:700;color:#FB7185;margin-bottom:0.5rem'>"
                        "🔴 Opposing the prediction</div>",
                        unsafe_allow_html=True
                    )
                    if neg_words:
                        max_s = max(s for _, s in neg_words) or 1
                        for word, score in neg_words:
                            bar_w = int(score / max_s * 100)
                            st.markdown(f"""
                            <div style='margin-bottom:0.4rem'>
                                <div style='display:flex;justify-content:space-between;margin-bottom:2px'>
                                    <span style='color:#F8FAFC;font-family:JetBrains Mono,monospace;font-size:.8rem'>{word}</span>
                                    <span style='color:#94a3b8;font-size:.72rem'>{score:.3f}</span>
                                </div>
                                <div class='timeline-bar'>
                                    <div class='timeline-fill' style='width:{bar_w}%;background:#FB7185'></div>
                                </div>
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color:#94A3B8;font-size:.82rem'>No opposing signals found.</span>", unsafe_allow_html=True)

        elif analyze_btn:
            st.warning("Please paste a review before analyzing.")

    with mode[1]:
        st.markdown("<div style='font-size:0.88rem;color:#94a3b8;margin-bottom:1rem'>Enter multiple reviews below (one review per line) to process in batch.</div>", unsafe_allow_html=True)
        
        batch_demo = st.toggle("Pre-fill with Demo Batch (12 reviews)", value=True, key="batch_demo_toggle")
        
        if batch_demo:
            batch_default = "\n".join([
                "This app is absolutely amazing! The UI is clean and everything loads fast. Love the dark mode.",
                "Keeps crashing every time I open it. Terrible experience. Uninstalling now.",
                "Pretty average app. Does what it says but nothing special. Decent performance.",
                "Battery drain is insane! My phone gets hot within 10 minutes of using this app.",
                "Best productivity app I've ever used. Smooth, intuitive, and well designed.",
                "Laggy interface, tons of bugs. The payment system doesn't work either. Scam!",
                "Works fine for basic needs. Notifications are a bit spammy though.",
                "Outstanding update! All previous issues fixed. Now runs perfectly on my device.",
                "Privacy is a huge concern. Too many permissions required. Not happy about data collection.",
                "Great app overall. Minor UI glitches but nothing that ruins the experience.",
                "Frozen twice today. Can't even load the main screen. Please fix ASAP.",
                "Love the new features in this update. The sync across devices is brilliant!"
            ])
        else:
            batch_default = ""
            
        batch_input = st.text_area(
            "Batch Text Input",
            value=batch_default,
            height=200,
            placeholder="Paste reviews here, one per line...",
            label_visibility="collapsed",
            key="batch_input_area"
        )
        
        run_batch = st.button("📋 Run Batch Classification", key="run_batch_btn")
        
        if run_batch and batch_input.strip():
            lines = [line.strip() for line in batch_input.split("\n") if line.strip()]
            if not lines:
                st.warning("No valid reviews found.")
            else:
                with st.spinner("Processing batch..."):
                    batch_df = classify_batch(lines)
                
                counts = batch_df["Sentiment"].value_counts()
                tot = len(batch_df)
                p_cnt = counts.get("Positive", 0)
                n_cnt = counts.get("Negative", 0)
                u_cnt = counts.get("Neutral", 0)
                
                # Stats
                st.markdown("<div class='section-label' style='margin-top:1.5rem'>Batch Performance Summary</div>", unsafe_allow_html=True)
                bc1, bc2, bc3, bc4 = st.columns(4)
                bc1.metric("Total Input", tot)
                bc2.metric("😊 Positive Ratio", f"{p_cnt} ({int(p_cnt/tot*100)}%)")
                bc3.metric("😞 Negative Ratio", f"{n_cnt} ({int(n_cnt/tot*100)}%)")
                bc4.metric("😐 Neutral Ratio", f"{u_cnt} ({int(u_cnt/tot*100)}%)")
                
                # Distribution bar
                pct_pos = p_cnt / tot
                pct_neg = n_cnt / tot
                pct_neu = u_cnt / tot
                st.markdown(f"""
                <div style='margin:1rem 0;border-radius:8px;overflow:hidden;height:12px;display:flex'>
                    <div style='width:{pct_pos*100:.1f}%;background:#34D399' title='Positive'></div>
                    <div style='width:{pct_neu*100:.1f}%;background:#A78BFA' title='Neutral'></div>
                    <div style='width:{pct_neg*100:.1f}%;background:#FB7185' title='Negative'></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Table
                st.markdown("<div class='section-label'>Analyzed Data</div>", unsafe_allow_html=True)
                def style_s(val):
                    if val == "Positive":
                        return "background-color: rgba(34,197,94,.12); color: #34D399; font-weight: bold;"
                    elif val == "Negative":
                        return "background-color: rgba(244,63,94,.12); color: #FB7185; font-weight: bold;"
                    return "background-color: rgba(245,158,11,.12); color: #A78BFA; font-weight: bold;"
                
                try:
                    styled_batch = batch_df.style.map(style_s, subset=["Sentiment"])
                except AttributeError:
                    styled_batch = batch_df.style.applymap(style_s, subset=["Sentiment"])
                    
                st.dataframe(
                    styled_batch,
                    use_container_width=True,
                    height=250
                )
        elif run_batch:
            st.warning("Please provide input text or toggle the demo batch.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Data Explorer":
    st.markdown("""
    <div style='padding:2rem 0 1rem 0'>
        <div class='section-label'>Review Dataset Management</div>
        <div style='font-family:Plus Jakarta Sans,sans-serif;font-size:1.8rem;font-weight:800;color:#F8FAFC'>
            Data Explorer
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Uploader section
    with st.expander("📤 Upload Custom Review Dataset (CSV)", expanded=(st.session_state["custom_data"] is None)):
        st.markdown("""
        <div style='font-size:0.85rem;color:#94a3b8;margin-bottom:1rem'>
            Upload a CSV file containing reviews. We will run sentiment classification on the selected review column, 
            extract issues, and let you explore or download the fully processed dataset.
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded CSV with {len(raw_df)} rows and columns: {list(raw_df.columns)}")
                
                # Column selector
                txt_col = st.selectbox("Select the column containing the review text", list(raw_df.columns))
                
                if st.button("🚀 Analyze Uploaded Dataset"):
                    with st.spinner("Classifying dataset sentiment & mapping issues..."):
                        records = []
                        start_d = datetime.now() - timedelta(days=30)
                        
                        devices = ["iPhone 15 Pro", "Samsung S24 Ultra", "Google Pixel 8 Pro", "OnePlus 12", "iPad Pro", "Samsung Galaxy Tab S9"]
                        versions = ["2.4.1", "2.4.0", "2.3.5", "2.2.0", "2.1.1", "2.0.0"]
                        names = ["Alex M.", "Sarah K.", "David L.", "Emily R.", "James P.", "Jessica T.", "Michael B.", "Sophia W.", "Daniel H.", "Olivia C.", "Chris T.", "Anna S."]
                        
                        for idx, row in raw_df.iterrows():
                            txt = str(row[txt_col])
                            
                            res_lr = analyze_sentiment(txt, "Logistic Regression")
                            res_nb = analyze_sentiment(txt, "Multinomial Naive Bayes")
                            res_rob = analyze_sentiment(txt, "RoBERTa (Transformer)")
                            
                            rating = 3
                            if "rating" in [c.lower() for c in raw_df.columns]:
                                rat_col = [c for c in raw_df.columns if c.lower() == "rating"][0]
                                try:
                                    rating = int(float(row[rat_col]))
                                except:
                                    pass
                            else:
                                if res_lr["sentiment"] == "Positive":
                                    rating = 5
                                elif res_lr["sentiment"] == "Negative":
                                    rating = 1
                            
                            dev = str(row["device"]) if "device" in [c.lower() for c in raw_df.columns] else random.choice(devices)
                            ver = str(row["version"]) if "version" in [c.lower() for c in raw_df.columns] else random.choice(versions)
                            name = str(row["username"]) if "username" in [c.lower() for c in raw_df.columns] else random.choice(names)
                            
                            date = start_d + timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
                            if "date" in [c.lower() for c in raw_df.columns]:
                                dat_col = [c for c in raw_df.columns if c.lower() == "date"][0]
                                try:
                                    date = pd.to_datetime(row[dat_col])
                                except:
                                    pass
                                    
                            records.append({
                                "Date": date,
                                "Username": name,
                                "Rating": rating,
                                "Device": dev,
                                "Version": ver,
                                "Review": txt,
                                
                                "LR Sentiment": res_lr["sentiment"],
                                "LR Confidence": res_lr["confidence"],
                                "LR Pos Score": res_lr["pos_score"],
                                "LR Neg Score": res_lr["neg_score"],
                                
                                "NB Sentiment": res_nb["sentiment"],
                                "NB Confidence": res_nb["confidence"],
                                "NB Pos Score": res_nb["pos_score"],
                                "NB Neg Score": res_nb["neg_score"],
                                
                                "RoBERTa Sentiment": res_rob["sentiment"],
                                "RoBERTa Confidence": res_rob["confidence"],
                                "RoBERTa Pos Score": res_rob["pos_score"],
                                "RoBERTa Neg Score": res_rob["neg_score"],
                                
                                "Issues": ", ".join(res_lr["detected_issues"].keys()) if res_lr["detected_issues"] else "None",
                            })
                            
                        processed_df = pd.DataFrame(records)
                        processed_df = processed_df.sort_values(by="Date", ascending=False).reset_index(drop=True)
                        st.session_state["custom_data"] = processed_df
                        st.session_state["dataset_selection"] = "Uploaded Dataset"
                        if hasattr(st, "rerun"):
                            st.rerun()
                        else:
                            st.experimental_rerun()
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")

    st.markdown(f"**Current Dataset:** `{st.session_state['dataset_selection']}` ({len(active_df)} reviews)", unsafe_allow_html=True)

    # Filters panel
    st.markdown("<div class='section-label'>Dataset Filters</div>", unsafe_allow_html=True)
    fcol1, fcol2, fcol3 = st.columns(3)
    
    with fcol1:
        search_query = st.text_input("🔍 Search Reviews", "", placeholder="Enter keyword...")
        sentiment_filter = st.selectbox("Sentiment", ["All", "Positive", "Negative", "Neutral"])
        
    with fcol2:
        rating_filter = st.multiselect("Rating Stars", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
        unique_devices = ["All"] + sorted(list(active_df["Device"].unique())) if not active_df.empty else ["All"]
        device_filter = st.selectbox("Device Model", unique_devices)
        
    with fcol3:
        unique_versions = ["All"] + sorted(list(active_df["Version"].unique())) if not active_df.empty else ["All"]
        version_filter = st.selectbox("App Version", unique_versions)
        
        cats_list = ["All", "Bugs & Errors", "Crashes & Freezes", "Performance", "Battery & Heat", "UI / Design", "Privacy & Security", "Payments", "Connectivity", "Notifications", "Features"]
        issue_filter = st.selectbox("Issue Category Tag", cats_list)

    # Apply filters
    filtered_df = active_df.copy()
    
    if search_query:
        filtered_df = filtered_df[filtered_df["Review"].str.contains(search_query, case=False, na=False)]
        
    if sentiment_filter != "All":
        filtered_df = filtered_df[filtered_df["Sentiment"] == sentiment_filter]
        
    if rating_filter:
        filtered_df = filtered_df[filtered_df["Rating"].isin(rating_filter)]
        
    if device_filter != "All":
        filtered_df = filtered_df[filtered_df["Device"] == device_filter]
        
    if version_filter != "All":
        filtered_df = filtered_df[filtered_df["Version"] == version_filter]
        
    if issue_filter != "All":
        clean_tag = issue_filter.lower()
        if "bugs" in clean_tag:
            tag_match = "Bugs"
        elif "crashes" in clean_tag:
            tag_match = "Crashes"
        elif "performance" in clean_tag:
            tag_match = "Performance"
        elif "battery" in clean_tag:
            tag_match = "Battery"
        elif "ui" in clean_tag:
            tag_match = "UI"
        elif "privacy" in clean_tag:
            tag_match = "Privacy"
        elif "payments" in clean_tag:
            tag_match = "Payments"
        elif "connectivity" in clean_tag:
            tag_match = "Connectivity"
        elif "notifications" in clean_tag:
            tag_match = "Notifications"
        elif "features" in clean_tag:
            tag_match = "Features"
        else:
            tag_match = issue_filter
            
        filtered_df = filtered_df[filtered_df["Issues"].str.contains(tag_match, case=False, na=False)]

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dataset statistics and distributions section
    st.markdown("<div class='section-label'>Dataset Statistics & Distributions</div>", unsafe_allow_html=True)
    total_reviews = len(filtered_df)
    if total_reviews > 0:
        avg_rating = filtered_df["Rating"].mean()
        avg_confidence = filtered_df["Confidence"].mean()
        pos_pct = (filtered_df["Sentiment"] == "Positive").sum() / total_reviews * 100
        neg_pct = (filtered_df["Sentiment"] == "Negative").sum() / total_reviews * 100
        neu_pct = (filtered_df["Sentiment"] == "Neutral").sum() / total_reviews * 100
        issues_count = (filtered_df["Issues"] != "None").sum()
    else:
        avg_rating = 0.0
        avg_confidence = 0.0
        pos_pct = 0.0
        neg_pct = 0.0
        neu_pct = 0.0
        issues_count = 0

    col1_dist, col2_dist, col3_dist = st.columns([4, 3, 3])
    
    with col1_dist:
        st.markdown(f"""
        <div class='card' style='padding: 1.2rem; height: 100%; min-height: 250px;'>
            <div style='font-family:Plus Jakarta Sans,sans-serif;font-size:0.95rem;font-weight:700;color:#F8FAFC;margin-bottom:0.6rem;'>Summary Statistics</div>
            <div style='display:flex; flex-direction:column; gap:0.5rem; color:#94a3b8; font-size:0.85rem;'>
                <div style='display:flex; justify-content:space-between; border-bottom:1px solid #2A3A50; padding-bottom:2px;'>
                    <span>Total Matches:</span>
                    <strong style='color:#F8FAFC'>{total_reviews} reviews</strong>
                </div>
                <div style='display:flex; justify-content:space-between; border-bottom:1px solid #2A3A50; padding-bottom:2px;'>
                    <span>Average User Rating:</span>
                    <strong style='color:#F8FAFC'>{avg_rating:.2f} ★</strong>
                </div>
                <div style='display:flex; justify-content:space-between; border-bottom:1px solid #2A3A50; padding-bottom:2px;'>
                    <span>Average Confidence:</span>
                    <strong style='color:#F8FAFC'>{avg_confidence:.1f}%</strong>
                </div>
                <div style='display:flex; justify-content:space-between; border-bottom:1px solid #2A3A50; padding-bottom:2px;'>
                    <span>Positive Reviews:</span>
                    <strong style='color:#34D399'>{pos_pct:.1f}% ({int((filtered_df["Sentiment"] == "Positive").sum())})</strong>
                </div>
                <div style='display:flex; justify-content:space-between; border-bottom:1px solid #2A3A50; padding-bottom:2px;'>
                    <span>Negative Reviews:</span>
                    <strong style='color:#FB7185'>{neg_pct:.1f}% ({int((filtered_df["Sentiment"] == "Negative").sum())})</strong>
                </div>
                <div style='display:flex; justify-content:space-between;'>
                    <span>Neutral Reviews:</span>
                    <strong style='color:#A78BFA'>{neu_pct:.1f}% ({int((filtered_df["Sentiment"] == "Neutral").sum())})</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2_dist:
        st.markdown("<div style='font-family:Plus Jakarta Sans,sans-serif;font-size:0.95rem;font-weight:700;color:#F8FAFC;margin-bottom:0.4rem;text-align:center;'>Sentiment Distribution (Pie)</div>", unsafe_allow_html=True)
        if total_reviews > 0:
            import altair as alt
            pie_data = pd.DataFrame({
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Count": [
                    int((filtered_df["Sentiment"] == "Positive").sum()),
                    int((filtered_df["Sentiment"] == "Neutral").sum()),
                    int((filtered_df["Sentiment"] == "Negative").sum())
                ]
            })
            pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=45).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Sentiment", type="nominal", scale=alt.Scale(
                    domain=["Positive", "Neutral", "Negative"],
                    range=["#34D399", "#A78BFA", "#FB7185"]
                ), legend=alt.Legend(title=None, orient="bottom", labelColor="#94a3b8")),
                tooltip=["Sentiment", "Count"]
            ).properties(
                height=180
            ).configure_view(
                strokeOpacity=0
            )
            st.altair_chart(pie_chart, use_container_width=True)
        else:
            st.info("No data matched.")
            
    with col3_dist:
        st.markdown("<div style='font-family:Plus Jakarta Sans,sans-serif;font-size:0.95rem;font-weight:700;color:#F8FAFC;margin-bottom:0.4rem;text-align:center;'>Rating Distribution (Bar)</div>", unsafe_allow_html=True)
        if total_reviews > 0:
            import altair as alt
            rating_counts = filtered_df["Rating"].value_counts().reindex([1, 2, 3, 4, 5]).fillna(0).reset_index()
            rating_counts.columns = ["Rating", "Count"]
            rating_counts["Rating"] = rating_counts["Rating"].astype(int).astype(str) + "★"
            
            bar_chart = alt.Chart(rating_counts).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("Rating:N", axis=alt.Axis(labelAngle=0, labelColor="#94a3b8", title=None)),
                y=alt.Y("Count:Q", axis=alt.Axis(labelColor="#94a3b8", title=None)),
                color=alt.value("#38BDF8"),
                tooltip=["Rating", "Count"]
            ).properties(
                height=180
            ).configure_view(
                strokeOpacity=0
            )
            st.altair_chart(bar_chart, use_container_width=True)
        else:
            st.info("No data matched.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem'>
        <div class='section-label'>Matched Records ({len(filtered_df)} / {len(active_df)})</div>
    </div>
    """, unsafe_allow_html=True)

    if filtered_df.empty:
        st.info("No reviews matched your filter settings. Try adjusting search or filters.")
    else:
        def style_sentiment_cells(val):
            if val == "Positive":
                return "background-color: rgba(34,197,94,.08); color: #34D399"
            elif val == "Negative":
                return "background-color: rgba(244,63,94,.08); color: #FB7185"
            return "background-color: rgba(245,158,11,.08); color: #A78BFA"
            
        try:
            styled_table = filtered_df.style.map(style_sentiment_cells, subset=["Sentiment"])
        except AttributeError:
            styled_table = filtered_df.style.applymap(style_sentiment_cells, subset=["Sentiment"])
        
        st.dataframe(styled_table, use_container_width=True, height=400)
        
        csv_buffer = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered CSV",
            data=csv_buffer,
            file_name=f"reviewlens_filtered_{st.session_state['dataset_selection'].lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Visualizations":
    st.markdown("""
    <div style='padding:2rem 0 1rem 0'>
        <div class='section-label'>Sentiment Analytics & Trends</div>
        <div style='font-family:Plus Jakarta Sans,sans-serif;font-size:1.8rem;font-weight:800;color:#F8FAFC'>
            Visualizations
        </div>
    </div>
    """, unsafe_allow_html=True)

    if active_df.empty:
        st.warning("No data available to build dashboard charts. Please upload a dataset in Data Explorer.")
    else:
        st.markdown("<div class='section-label'>Date Filter Selection</div>", unsafe_allow_html=True)
        min_date = active_df["Date"].min()
        max_date = active_df["Date"].max()
        
        if pd.isnull(min_date) or pd.isnull(max_date) or min_date == max_date:
            min_date = datetime.now() - timedelta(days=30)
            max_date = datetime.now()
            
        date_range = st.slider(
            "Select Time Frame",
            min_value=min_date.to_pydatetime(),
            max_value=max_date.to_pydatetime(),
            value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
            format="MMM DD",
            label_visibility="collapsed"
        )
        
        vis_df = active_df[(active_df["Date"] >= date_range[0]) & (active_df["Date"] <= date_range[1])]
        
        if vis_df.empty:
            st.info("No reviews exist in the selected time range.")
        else:
            st.markdown("<div class='section-label' style='margin-top:1rem'>Performance KPIs</div>", unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            
            p_count = (vis_df["Sentiment"] == "Positive").sum()
            n_count = (vis_df["Sentiment"] == "Negative").sum()
            total_n = len(vis_df)
            
            nsi = ((p_count - n_count) / total_n * 100) if total_n > 0 else 0
            
            k1.metric("Average Rating", f"{vis_df['Rating'].mean():.2f} ★")
            k2.metric("Net Sentiment Index (NSI)", f"{nsi:+.1f}%")
            k3.metric("Review Volume", f"{total_n}")
            
            all_issues_list = []
            for issues_str in vis_df["Issues"].dropna():
                if issues_str != "None":
                    all_issues_list.extend([i.strip() for i in issues_str.split(",")])
            
            issue_freq = Counter(all_issues_list)
            top_issue = issue_freq.most_common(1)[0][0] if issue_freq else "None"
            k4.metric("Top Complaint Area", top_issue)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            chart_tabs = st.tabs([
                "📈 Sentiment Trend Over Time", 
                "📊 Sentiment & Rating Distribution", 
                "🐛 Product Issues Breakdown", 
                "☁️ Word Cloud", 
                "🎯 Confusion Matrix Heatmap", 
                "⚔️ Model Comparison"
            ])
            
            with chart_tabs[0]:
                trend_df = get_trend_data_from_df(vis_df)
                if not trend_df.empty:
                    chart_data = trend_df.set_index("Date")[["Positive", "Negative", "Neutral"]]
                    st.line_chart(
                        chart_data,
                        color=["#34D399", "#FB7185", "#A78BFA"],
                        height=300,
                        use_container_width=True,
                    )
                else:
                    st.info("Insufficient data points to plot trend line.")
                    
            with chart_tabs[1]:
                col_dist1, col_dist2 = st.columns(2)
                with col_dist1:
                    st.markdown("<div style='font-family:Plus Jakarta Sans,sans-serif;font-size:1rem;font-weight:700;color:#F8FAFC;margin-bottom:0.8rem'>Sentiment Label Distribution</div>", unsafe_allow_html=True)
                    label_counts = vis_df["Sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
                    st.bar_chart(label_counts, color="#38BDF8", height=280, use_container_width=True)
                with col_dist2:
                    st.markdown("<div style='font-family:Plus Jakarta Sans,sans-serif;font-size:1rem;font-weight:700;color:#F8FAFC;margin-bottom:0.8rem'>Star Rating Distribution</div>", unsafe_allow_html=True)
                    rating_counts = vis_df["Rating"].value_counts().reindex([1, 2, 3, 4, 5]).fillna(0)
                    st.bar_chart(rating_counts, color="#34D399", height=280, use_container_width=True)
                    
            with chart_tabs[2]:
                if issue_freq:
                    st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)
                    max_freq = max(issue_freq.values())
                    for issue_tag, count in sorted(issue_freq.items(), key=lambda x: -x[1]):
                        pct = count / total_n * 100
                        fill_width = int(count / max_freq * 100)
                        
                        bar_color = "#FB7185" if "Bug" in issue_tag or "Crash" in issue_tag or "Performance" in issue_tag else "#38BDF8"
                        
                        st.markdown(f"""
                        <div style='margin-bottom:.8rem'>
                            <div style='display:flex;justify-content:space-between;margin-bottom:4px'>
                                <span style='color:#F8FAFC;font-size:.88rem;font-weight:500'>{issue_tag}</span>
                                <span style='color:#94A3B8;font-size:.78rem'>{count} occurrences ({pct:.1f}%)</span>
                            </div>
                            <div class='timeline-bar'>
                                <div class='timeline-fill' style='width:{fill_width}%;background:{bar_color}'></div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info("No tagged complaints/issues detected in the current subset.")

            with chart_tabs[3]:
                st.markdown("<div style='font-family:Plus Jakarta Sans,sans-serif;font-size:1.1rem;font-weight:700;color:#F8FAFC;margin-bottom:0.8rem'>Word Cloud of Most Common Words</div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:0.85rem;color:#94a3b8;margin-bottom:1rem'>Displays the most frequent words in reviews. Green indicates positive lexicon, red indicates negative lexicon, and other colors represent general terms. Hover for word frequency.</div>", unsafe_allow_html=True)
                
                STOPWORDS = {"the", "and", "a", "of", "to", "is", "it", "in", "this", "app", "i", "for", "with", "my", "on", "but", "have", "you", "that", "so", "be", "was", "are", "not", "but", "as", "at", "or", "an", "they", "we", "just", "if", "from", "very", "about", "all", "out", "can", "here", "too", "has", "when", "more", "would", "get", "like", "use", "only", "even", "does", "been", "than"}
                
                all_words = []
                for review in vis_df["Review"].dropna():
                    cleaned_r = clean_text(review)
                    for w in cleaned_r.split():
                        if len(w) > 2 and w not in STOPWORDS:
                            all_words.append(w)
                
                word_counts = Counter(all_words).most_common(40)
                
                if word_counts:
                    max_c = max(w[1] for w in word_counts)
                    min_c = min(w[1] for w in word_counts)
                    spread = max_c - min_c if max_c != min_c else 1
                    
                    html_cloud = "<div style='display:flex; flex-wrap:wrap; justify-content:center; align-items:center; gap:16px; padding:25px; background:#141C2E; border:1px solid #2A3A50; border-radius:12px; min-height:220px;'>"
                    
                    colors_palette = ["#38BDF8", "#38bdf8", "#818cf8", "#a78bfa", "#A78BFA", "#94a3b8", "#F8FAFC"]
                    
                    for word, count in word_counts:
                        font_size = 13 + int((count - min_c) / spread * 22)
                        
                        color_idx = sum(ord(char) for char in word) % len(colors_palette)
                        color = colors_palette[color_idx]
                        
                        html_cloud += f"<span style='font-family:Plus Jakarta Sans, sans-serif; font-size:{font_size}px; color:{color}; font-weight:700; padding:4px 8px; cursor:default;' title='Frequency: {count}'>{word}</span>"
                    
                    html_cloud += "</div>"
                    st.markdown(html_cloud, unsafe_allow_html=True)
                else:
                    st.info("Insufficient words found to build word cloud.")

            with chart_tabs[4]:
                st.markdown("<div style='font-family:Plus Jakarta Sans,sans-serif;font-size:1.1rem;font-weight:700;color:#F8FAFC;margin-bottom:0.8rem'>Confusion Matrix Heatmap</div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:0.85rem;color:#94a3b8;margin-bottom:1rem'>Compares actual user ratings (Ground Truth: 1-2★ as Negative, 3★ as Neutral, 4-5★ as Positive) with the selected model predictions. Diagonal elements represent correct classifications.</div>", unsafe_allow_html=True)
                
                # Compute Confusion Matrix
                actual_classes = ["Positive", "Neutral", "Negative"]
                pred_classes = ["Positive", "Neutral", "Negative"]
                
                conf_matrix = {a: {p: 0 for p in pred_classes} for a in actual_classes}
                
                for _, row in vis_df.iterrows():
                    rating = row["Rating"]
                    act = "Positive" if rating >= 4 else ("Negative" if rating <= 2 else "Neutral")
                    prd = row["Sentiment"]
                    if act in conf_matrix and prd in conf_matrix[act]:
                        conf_matrix[act][prd] += 1
                
                # Render Confusion Matrix as HTML Heatmap Table
                html_matrix = """
                <table style='width:100%; border-collapse:collapse; background:#141C2E; border:1px solid #2A3A50; border-radius:12px; overflow:hidden;'>
                    <thead>
                        <tr style='background:#1E293B; border-bottom:1px solid #2A3A50; text-align:center;'>
                            <th style='padding:12px; color:#F8FAFC; font-family:Plus Jakarta Sans,sans-serif; font-size:0.85rem; text-align:left;'>Actual \\ Predicted</th>
                            <th style='padding:12px; color:#34D399; font-family:Plus Jakarta Sans,sans-serif; font-size:0.85rem;'>Predicted Positive</th>
                            <th style='padding:12px; color:#A78BFA; font-family:Plus Jakarta Sans,sans-serif; font-size:0.85rem;'>Predicted Neutral</th>
                            <th style='padding:12px; color:#FB7185; font-family:Plus Jakarta Sans,sans-serif; font-size:0.85rem;'>Predicted Negative</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for act in actual_classes:
                    act_color = "#34D399" if act == "Positive" else ("#FB7185" if act == "Negative" else "#A78BFA")
                    html_matrix += f"<tr style='border-bottom:1px solid #2A3A50; text-align:center;'>"
                    html_matrix += f"<td style='padding:14px; font-weight:bold; color:{act_color}; background:#1E293B; font-size:0.85rem; text-align:left;'>Actual {act}</td>"
                    
                    total_act = sum(conf_matrix[act].values())
                    
                    for prd in pred_classes:
                        count = conf_matrix[act][prd]
                        pct = (count / total_act * 100) if total_act > 0 else 0
                        
                        if act == prd:
                            bg_c = f"rgba(34, 197, 94, {0.05 + (pct/100)*0.35})"
                            text_c = "#34D399"
                            font_w = "bold"
                        else:
                            if count > 0:
                                bg_c = f"rgba(244, 63, 94, {0.05 + (pct/100)*0.35})"
                                text_c = "#FB7185"
                                font_w = "normal"
                            else:
                                bg_c = "transparent"
                                text_c = "#94A3B8"
                                font_w = "normal"
                                
                        html_matrix += f"<td style='padding:14px; background:{bg_c}; color:{text_c}; font-weight:{font_w}; font-size:0.9rem;'>"
                        html_matrix += f"{count}<br><span style='font-size:0.75rem; color:#94a3b8;'>({pct:.1f}%)</span>"
                        html_matrix += "</td>"
                    html_matrix += "</tr>"
                    
                html_matrix += "</tbody></table>"
                
                col_mat_l, col_mat_r = st.columns([3, 2])
                with col_mat_l:
                    st.markdown(html_matrix, unsafe_allow_html=True)
                with col_mat_r:
                    st.markdown("<div style='font-family:Plus Jakarta Sans,sans-serif;font-size:1rem;font-weight:700;color:#F8FAFC;margin-bottom:0.8rem'>Confidence Metrics</div>", unsafe_allow_html=True)
                    conf_by_rating = vis_df.groupby("Rating")["Confidence"].mean().reindex([1, 2, 3, 4, 5]).fillna(0)
                    st.bar_chart(conf_by_rating, color="#A78BFA", height=220, use_container_width=True)

            with chart_tabs[5]:
                st.markdown("<div style='font-family:Plus Jakarta Sans,sans-serif;font-size:1.1rem;font-weight:700;color:#F8FAFC;margin-bottom:0.8rem'>Model Comparison Chart</div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:0.85rem;color:#94a3b8;margin-bottom:1rem'>Visualizing validation accuracy vs. inference speed trade-offs across different model families.</div>", unsafe_allow_html=True)
                
                col_comp_l, col_comp_r = st.columns(2)
                with col_comp_l:
                    st.markdown("<div style='font-size:0.85rem;color:#F8FAFC;margin-bottom:0.4rem;font-weight:bold'>Real-time Accuracy on Current Dataset (%) — Higher is Better</div>", unsafe_allow_html=True)
                    accuracy_df = pd.DataFrame({
                        "Accuracy (%)": [
                            realtime_accuracies["Multinomial Naive Bayes"],
                            realtime_accuracies["Logistic Regression"],
                            realtime_accuracies["RoBERTa (Transformer)"]
                        ]
                    }, index=["Multinomial Naive Bayes", "Logistic Regression", "RoBERTa (Transformer)"])
                    st.bar_chart(accuracy_df, color="#FBBF24", height=250, use_container_width=True)
                with col_comp_r:
                    st.markdown("<div style='font-size:0.85rem;color:#F8FAFC;margin-bottom:0.4rem;font-weight:bold'>Measured Latency per Review (ms) — Lower is Better</div>", unsafe_allow_html=True)
                    latency_df = pd.DataFrame({
                        "Latency (ms)": [
                            st.session_state["latencies"]["Multinomial Naive Bayes"],
                            st.session_state["latencies"]["Logistic Regression"],
                            st.session_state["latencies"]["RoBERTa (Transformer)"]
                        ]
                    }, index=["Multinomial Naive Bayes", "Logistic Regression", "RoBERTa (Transformer)"])
                    st.bar_chart(latency_df, color="#FB7185", height=250, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ Model Info":
    st.markdown("""
    <div style='padding:2rem 0 1rem 0'>
        <div class='section-label'>NLP Classifier Architecture</div>
        <div style='font-family:Plus Jakarta Sans,sans-serif;font-size:1.8rem;font-weight:800;color:#F8FAFC'>
            Model Info
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <div class='section-label'>Model Summary</div>
        <div style='font-family:Plus Jakarta Sans,sans-serif;font-size:1.15rem;font-weight:700;color:#F8FAFC;margin-bottom:.5rem'>
            Machine Learning & Transformer Architectures
        </div>
        <div style='color:#94a3b8;font-size:0.9rem;line-height:1.7'>
            ReviewLens runs on a state-of-the-art NLP classification pipeline. We combine fast probabilistic linear models with a heavy transformer-based encoder model to provide a robust tradeoff between speed, explainability, and deep semantic accuracy.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Processing Pipeline Architecture</div>", unsafe_allow_html=True)
    with st.expander("🛠️ Step-by-Step Processing Flow", expanded=True):
        st.markdown("""
        <div style='color:#94a3b8;font-size:.88rem;line-height:1.8;padding:.2rem 0'>
        <b style='color:#F8FAFC'>1. TF-IDF + Machine Learning Pipeline (Logistic Regression & Naive Bayes):</b> <br>
        • <i>Text Preprocessing:</i> Lowercasing, removal of punctuation, and tokenization. <br>
        • <i>Vectorization:</i> Term Frequency-Inverse Document Frequency (TF-IDF) representation converts text into numeric vectors (unigrams & bigrams). <br>
        • <i>Classification:</i> The probability distribution over classes is calculated based on model weights (Logistic Regression coefficients or Naive Bayes priors). <br><br>
        <b style='color:#F8FAFC'>2. RoBERTa Transformer Pipeline:</b> <br>
        • <i>Subword Tokenization:</i> The review is tokenized using Byte-Pair Encoding (BPE). <br>
        • <i>Attention Mechanism:</i> Bidirectional self-attention layers compute contextual representations for each word. <br>
        • <i>Classification Head:</i> A softmax head processes the pooled output of the encoder to calculate exact probabilities for Positive, Neutral, and Negative sentiments.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-label' style='margin-top:1.5rem'>Model Evaluation & Tuning</div>", unsafe_allow_html=True)
    metric_tabs = st.tabs(["⚙️ Model Benchmarks", "🧪 Training Details", "🐛 Product Issue Categories"])
    
    with metric_tabs[0]:
        st.markdown("""
        <div style='color:#94a3b8;font-size:0.9rem;line-height:1.7;margin-bottom:1rem'>
            Comparative benchmark of our model architectures under standard production workloads:
        </div>
        """, unsafe_allow_html=True)
        
        comparison_data = {
            "Architecture Type": ["Multinomial Naive Bayes", "Logistic Regression + TF-IDF", "RoBERTa (Transformer-based)"],
            "Real-time Accuracy (Current Dataset)": [
                f"{realtime_accuracies['Multinomial Naive Bayes']:.1f}%",
                f"{realtime_accuracies['Logistic Regression']:.1f}%",
                f"{realtime_accuracies['RoBERTa (Transformer)']:.1f}%"
            ],
            "Measured Latency (per review)": [
                f"{st.session_state['latencies']['Multinomial Naive Bayes']:.2f} ms",
                f"{st.session_state['latencies']['Logistic Regression']:.2f} ms",
                f"{st.session_state['latencies']['RoBERTa (Transformer)']:.2f} ms"
            ],
            "Memory / RAM Footprint": ["~ 15.0 MB", "~ 45.0 MB", "~ 500.0 MB"],
            "Hosting Overhead / Cost": ["Low ($0 - Local)", "Low ($0 - Local)", "Medium (CPU Inference)"],
            "Explainability Level": ["High (Priors)", "High (Coefficients)", "Attention Maps Only"]
        }
        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with metric_tabs[1]:
        st.markdown("""
        <div class='card'>
            <div class='section-label'>Model Calibration Details</div>
            <div style='font-family:Plus Jakarta Sans,sans-serif;font-size:1.15rem;font-weight:700;color:#F8FAFC;margin-bottom:.5rem'>
                How our Classification Models Were Calibrated
            </div>
            <div style='color:#94a3b8;font-size:0.88rem;line-height:1.8'>
                1. <b>TF-IDF Vectorizer Calibration:</b> Vocabulary built from 50,000 mobile reviews, ignoring terms with document frequency below 5 or above 80% to filter out noise. <br>
                2. <b>Linear Model Training:</b> Multinomial Naive Bayes and Logistic Regression were trained on standard L2 regularization with grid search to identify optimal hyperparameter <i>C = 1.0</i> for Logistic Regression. <br>
                3. <b>RoBERTa Fine-tuning:</b> Based on the <i>cardiffnlp/twitter-roberta-base-sentiment-latest</i> checkpoint. The model was fine-tuned on task-specific review text using AdamW optimizer with a learning rate of 2e-5 and early stopping validation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with metric_tabs[2]:
        st.markdown("<div style='font-size:0.85rem;color:#94a3b8;margin-bottom:0.8rem'>Product issue categories mapping list (keyword-based category tagging):</div>", unsafe_allow_html=True)
        for cat, keywords in ISSUE_CATEGORIES.items():
            kw_tags = "".join([f"<span style='background:#1E293B;border:1px solid #2A3A50;border-radius:4px;padding:2px 8px;font-size:0.75rem;color:#F8FAFC;margin:2px;display:inline-block'>{k}</span>" for k in keywords])
            st.markdown(f"""
            <div style='background:#141C2E;border:1px solid #2A3A50;border-radius:12px;padding:12px;margin-bottom:8px'>
                <div style='font-size:0.88rem;font-weight:bold;color:#38BDF8;margin-bottom:6px'>{cat}</div>
                <div>{kw_tags}</div>
            </div>
            """, unsafe_allow_html=True)
