# app.py
import streamlit as st
import pickle
import re
from bs4 import BeautifulSoup
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# --- Pre-load NLTK data (for deployment environment) ---
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

# --- Load Saved Model & Vectorizer ---
try:
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    model = pickle.load(open('lr_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please run the training script to create them.")
    st.stop()

# --- Preprocessing Function (Must be identical to the one used in training) ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    tokens = word_tokenize(text)
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return " ".join(clean_tokens)

# --- Streamlit Web App Interface ---
st.set_page_config(page_title="Risk Indicator", page_icon="⚠️")

st.title("Text-Based Suicide Risk Indicator ⚠️")
st.markdown(
    "This application uses a **Logistic Regression** model to identify linguistic patterns "
    "that may be associated with suicide risk. Please enter a text post below."
)

st.error(
    "**Disclaimer:** This is an educational proof-of-concept and **NOT a diagnostic tool.** "
    "It is not a substitute for professional medical advice, diagnosis, or treatment. "
    "If you or someone you know is in crisis, please seek help immediately from a qualified professional."
)

user_input = st.text_area("Enter text for analysis:", "", height=150)

if st.button("Analyze Text"):
    if user_input:
        # 1. Preprocess, 2. Vectorize, 3. Predict
        cleaned_input = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]
        prediction_proba = model.predict_proba(vectorized_input)[0]

        st.subheader("Analysis Result")
        if prediction == 1:
            st.warning("The text shows linguistic patterns that may indicate suicide risk.")
            st.write(f"Model Confidence: {prediction_proba[1]*100:.2f}%")
        else:
            st.success("The text does not show linguistic patterns typically associated with suicide risk.")
            st.write(f"Model Confidence: {prediction_proba[0]*100:.2f}%")
    else:
        st.info("Please enter some text before clicking 'Analyze Text'.")