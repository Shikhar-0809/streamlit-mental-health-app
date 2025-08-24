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

# --- PAGE CONFIGURATION ---
# This must be the first Streamlit command.
st.set_page_config(
    page_title="Text-Based Risk Indicator",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- CUSTOM CSS FOR STYLING ---
# We'll inject some CSS to make the app look like the reference website.
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# You can create a style.css file or inject it directly like this:
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #f0f2f6; /* Light grey background */
    }

    /* Main content area */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Style for the main title */
    h1 {
        color: #1E3A8A; /* A deep blue color */
        font-weight: 600;
    }

    /* Style for section headers */
    h2 {
        color: #3B82F6; /* A brighter blue */
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 5px;
        margin-top: 2rem;
    }
    
    /* Custom button style */
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #3B82F6;
        background-color: #3B82F6;
        color: white;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: white;
        color: #3B82F6;
        border: 1px solid #3B82F6;
    }

    /* Add a subtle animation to containers */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .st-emotion-cache-1r4qj8v, .st-emotion-cache-0 { /* Target Streamlit's container classes */
        animation: fadeIn 0.5s ease-out;
    }
    
</style>
""", unsafe_allow_html=True)


# --- NLTK Data Download ---
# This is a more robust way to ensure NLTK data is available in the deployment environment.
# --- NLTK Data Download ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab/english')
except LookupError:
    nltk.download('punkt_tab')   # <--- Add this for NLTK >=3.8
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


# --- LOAD MODEL AND VECTORIZER ---
@st.cache_resource
def load_model_assets():
    try:
        vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
        model = pickle.load(open('lr_model.pkl', 'rb'))
        return vectorizer, model
    except FileNotFoundError:
        return None, None
vectorizer, model = load_model_assets()


# --- PREPROCESSING FUNCTION ---
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


# --- HEADER SECTION ---
st.title("Text-Based Suicide Risk Indicator ⚠️")
st.markdown(
    "This application uses a Machine Learning model to identify linguistic patterns "
    "that may be associated with suicide risk. This is an educational proof-of-concept."
)
st.warning(
    "**Disclaimer:** This is **NOT a diagnostic tool.** "
    "If you or someone you know is in crisis, please seek help immediately from a qualified professional."
)


# --- INTERACTIVE ANALYZER SECTION ---
if model is None or vectorizer is None:
    st.error("Model files not found. The app cannot function. Please ensure 'lr_model.pkl' and 'tfidf_vectorizer.pkl' are in the repository.")
else:
    with st.container():
        st.markdown("## Enter Text for Analysis")
        user_input = st.text_area(" ", "", height=150, placeholder="Type or paste a social media post here...")

        if st.button("Analyze Text"):
            if user_input:
                # 1. Preprocess, 2. Vectorize, 3. Predict
                cleaned_input = preprocess_text(user_input)
                vectorized_input = vectorizer.transform([cleaned_input])
                prediction = model.predict(vectorized_input)[0]
                prediction_proba = model.predict_proba(vectorized_input)[0]

                st.subheader("Analysis Result")
                if prediction == 1:
                    st.warning(f"**Result:** The text may contain patterns associated with suicide risk. (Confidence: {prediction_proba[1]*100:.2f}%)")
                else:
                    st.success(f"**Result:** The text does not appear to contain patterns associated with suicide risk. (Confidence: {prediction_proba[0]*100:.2f}%)")
            else:
                st.info("Please enter some text before clicking 'Analyze Text'.")


# --- PROJECT INFORMATION SECTIONS ---
st.markdown("---") # Visual separator

col1, col2 = st.columns(2)

with col1:
    st.markdown("## 🎯 Problem Statement")
    st.write(
        """
        The project aims to address the critical challenge of identifying potential suicide risk from social media text. 
        By leveraging Natural Language Processing (NLP) and Machine Learning, we seek to build an automated tool that can 
        classify text as either indicating potential risk or being normal, providing a first line of proactive detection.
        """
    )

with col2:
    st.markdown("## 📊 Dataset Used")
    st.write(
        """
        This model was trained on the **"Mental Health Social Media"** dataset from Kaggle. 
        It contains thousands of posts from Twitter, with a binary label indicating `1` for potential suicide risk or `0` for normal.
        A key challenge in this dataset was the significant class imbalance, which was addressed using the SMOTE technique during training.
        """
    )

st.markdown("## 🤖 About the Model")
st.write(
    """
    The core of this application is a **Logistic Regression** model. Here's a summary of the methodology:
    - **Text Preprocessing:** Cleaned text by removing URLs, mentions, and stopwords, followed by lemmatization.
    - **Feature Engineering:** Used TF-IDF to convert text into numerical vectors.
    - **Model Choice:** Logistic Regression was chosen for its strong baseline performance and high interpretability, which is crucial for a sensitive application like this. It provides a clear, understandable model for this classification task.
    - **Evaluation:** The model was evaluated on metrics like Accuracy, Precision, Recall, and F1-Score, with a particular focus on Recall to minimize false negatives.
    """
)