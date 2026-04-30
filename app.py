import streamlit as st
import torch
import tensorflow as tf
import pandas as pd
import numpy as np
import re
import string
import joblib
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer

# =========================
# LOAD PREPROCESSORS
# =========================
preprocessor_en = joblib.load("preprocessor_r_en.pkl")
preprocessor_ar = joblib.load("preprocessor_r_ar.pkl")

# =========================
# TEXT PROCESSING
# =========================
def process_review_text_en(text):
    if not isinstance(text, str):
        return pd.Series([np.nan, 0, 0])

    RLength = len(text)
    SentenceLength = len(text.split())

    stop_words = set(stopwords.words("english"))
    punctuations = set(string.punctuation)
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text.lower())
    processed_tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and t not in punctuations
    ]

    cleaned_text = " ".join(processed_tokens)
    return pd.Series([cleaned_text, RLength, SentenceLength])


def process_review_text_ar(text):
    if not isinstance(text, str):
        return pd.Series([np.nan, 0, 0])

    RLength = len(text)
    SentenceLength = len(text.split())

    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)

    stop_words = set(stopwords.words("arabic"))
    punctuations = set(string.punctuation)
    stemmer_ar = ISRIStemmer()

    tokens = word_tokenize(text.lower())
    processed_tokens = [
        stemmer_ar.stem(t)
        for t in tokens
        if t not in stop_words and t not in punctuations
    ]

    cleaned_text = " ".join(processed_tokens)
    return pd.Series([cleaned_text, RLength, SentenceLength])

# =========================
# MODEL
# =========================
class ANN(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# =========================
# LANGUAGE DETECTION
# =========================
def detect_language(text):
    arabic = len(re.findall(r'[\u0600-\u06FF]', text))
    english = len(re.findall(r'[A-Za-z]', text))
    return "english" if english > arabic else "arabic"

# =========================
# FEATURE ENGINEERING
# =========================
def build_features(text, lang):
    text = pd.Series([text])

    Numbers = text.str.count(r"\d")
    Special = text.str.count(r"[!@#$%^&*()]")

    if lang == "english":
        processed = text.apply(process_review_text_en)
        df = pd.DataFrame({
            "Numbers": Numbers,
            "Special": Special,
            "r_en": processed[0],
            "RLength": processed[1],
            "SentenceLength": processed[2]
        })
        df = preprocessor_en.transform(df)

    else:
        processed = text.apply(process_review_text_ar)
        df = pd.DataFrame({
            "Numbers": Numbers,
            "Special": Special,
            "r_ar": processed[0],
            "RLength": processed[1],
            "SentenceLength": processed[2]
        })
        df = preprocessor_ar.transform(df)

    if hasattr(df, "toarray"):
        df = df.toarray()

    return df

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    models = {}

    # TensorFlow
    models["tf_en"] = tf.keras.models.load_model("tf_model/tf_model_en.keras", compile=False)
    models["tf_ar"] = tf.keras.models.load_model("tf_model/tf_model_ar.keras", compile=False)

    # PyTorch
    models["torch_en"] = ANN(15004)
    models["torch_en"].load_state_dict(torch.load("torch_model/model_english.pth", map_location="cpu"))
    models["torch_en"].eval()

    models["torch_ar"] = ANN(10004)
    models["torch_ar"].load_state_dict(torch.load("torch_model/model_arabic.pth", map_location="cpu"))
    models["torch_ar"].eval()

    return models

models = load_models()

# =========================
# UI
# =========================
st.title("🧠 Multi-Language Sentiment AI")

framework = st.selectbox("Choose Framework", ["TensorFlow", "PyTorch"])
text = st.text_area("Enter your text")

# =========================
# PREDICTION
# =========================
if st.button("Predict"):

    if not text.strip():
        st.warning("Please enter text")
    else:

        with st.spinner("🤖 AI is analyzing your text..."):
            lang = detect_language(text)
            features = build_features(text, lang)

            # =====================
            # TensorFlow
            # =====================
            if framework == "TensorFlow":

                model = models["tf_en"] if lang == "english" else models["tf_ar"]
                prob = model.predict(features)[0][0]

            # =====================
            # PyTorch
            # =====================
            else:

                tensor = torch.tensor(features, dtype=torch.float32)

                model = models["torch_en"] if lang == "english" else models["torch_ar"]

                with torch.no_grad():
                    prob = model(tensor).item()

        # =====================
        # RESULTS
        # =====================
        pos = prob
        neg = 1 - prob

        label = "Positive 😊" if prob > 0.5 else "Negative 😞"

        st.markdown("## 🎯 Result")
        st.success(f"{label}")

        # Probabilities
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Positive", f"{pos*100:.2f}%")

        with col2:
            st.metric("Negative", f"{neg*100:.2f}%")

        # Animation bar
        st.progress(int(pos * 100))

        # Interpretation
        if pos > 0.8:
            st.info("🔥 Very confident positive sentiment")
        elif pos > 0.6:
            st.info("🙂 Positive sentiment")
        elif pos > 0.4:
            st.warning("😐 Neutral / uncertain")
        else:
            st.error("⚠️ Strong negative sentiment")