pip install streamlit pymupdf python-docx scikit-learn nltk
%%writefile app.py
import streamlit as st
import numpy as np
import pandas as pd
import re
import fitz  # PyMuPDF for PDF extraction
from docx import Document
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset and train model
news_dataset = pd.read_csv('cleaned_train.csv')
news_dataset.dropna(inplace=True)
news_dataset['content'] = news_dataset['author'].fillna('') + ' ' + news_dataset['title'].fillna('')

# Stemming function
port_stem = PorterStemmer()
def stemming(content):
    content = re.sub(r'[^A-Za-z]', ' ', content).lower().split()
    return ' '.join([port_stem.stem(word) for word in content if word not in stop_words])

news_dataset['content'] = news_dataset['content'].apply(stemming)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(news_dataset['content'])
Y = news_dataset['label'].values

# Train model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
model = LogisticRegression()
model.fit(X_train, Y_train)

# Save Model & Vectorizer
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Load Model & Vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Function to process PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text("text") + " "
    return text

# Function to process DOCX
def extract_text_from_docx(uploaded_file):
    text = ""
    doc = Document(uploaded_file)
    for para in doc.paragraphs:
        text += para.text + " "
    return text

# Function to predict news authenticity
def predict_news(text):
    processed_text = stemming(text)
    transformed_text = vectorizer.transform([processed_text])
    prediction = model.predict(transformed_text)
    confidence = model.predict_proba(transformed_text).max() * 100
    return prediction[0], confidence

# Streamlit UI
st.set_page_config(page_title="Newx - Fake News Detector", page_icon="📰", layout="wide")
st.title("📰 Newx - Fake News Detection Web App")
st.markdown("**Upload a PDF or DOCX file containing a news article to check if it's Real or Fake.**")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        article_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        article_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Invalid file format. Please upload a PDF or DOCX.")
        article_text = ""
    
    if article_text.strip():
        st.write("### Extracted Text Preview:")
        st.text_area("Extracted Text", article_text[:1000], height=200)

        # Prediction
        result, confidence = predict_news(article_text)
        if result == 0:
            st.success(f"✅ The news is **Real** with {confidence:.2f}% confidence.")
        else:
            st.error(f"❌ The news is **Fake** with {confidence:.2f}% confidence.")
    else:
        st.warning("No text extracted. Please upload a valid document.")

# Adjustments for GitHub Codespaces
if __name__ == "__main__":
    st.write("Running on GitHub Codespaces...")
    st.write("Use the forwarded port to access the app.")
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
