# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 13:18:01 2024

@author: Talal Hassan Khan
"""

import streamlit as st
import pandas as pd
import re
import fitz  # PyMuPDF
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import os
import numpy as np

# Ensure the necessary NLTK data is downloaded
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model
model_filename = 'C:/Users/PMLS/Downloads/Hyperperameter_on_SpamEmail_Classifier/Model/best_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the vectorizer
vectorizer_filename = 'C:/Users/PMLS/Downloads/Hyperperameter_on_SpamEmail_Classifier/Model/vectorizer.pkl'
with open(vectorizer_filename, 'rb') as file:
    vectorizer = pickle.load(file)

# Streamlit app configuration
st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="wide")

# Add custom CSS for styling
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #008080 0%, #20B2AA 100%);
            padding: 10px;
            border-radius: 10px;
        }
        body {
            background-color: #1e1e1e;
            font-family: 'Helvetica Neue', sans-serif;
            color: #f0f0f0;
        }
        .container {
            background-color: #FFFFFF;
            color: #008080;
            font-size: 54px;
            font-weight: bold;
            border-radius: 10px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
            padding: 40px;
            margin: 20px auto;
            max-width: 1600px;
            text-align:center;
        }
        .title {
            background-color: #FFFFFF;
            color: #008080;
            font-size: 48px;
            font-weight: bold;
            margin-bottom: 20px;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);

        }
        .subheader {
            color: #FFFFFF;
            font-size: 24px;
            text-align: center;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #008080;
            color: white;
            border-radius: 10px;
            font-size: 20px;
            padding: 10px;
            border: none;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #006666;
        }
        .stTextArea>div>textarea, .stFileUploader>div>div {
            background-color: #333333;
            color: #f0f0f0;
            border: 2px solid #008080;
            border-radius: 10px;
            font-size: 22px;
            padding: 10px;
            margin-bottom: 20px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);

        }
        .stFileUploader>label {
            color: #f0f0f0;
        }
        .footer {
            color: #2c3e50;
            text-align: center;
            margin-top: 50px;
            font-size: 18px;
        }
        /* General selector for Streamlit file uploader label */
        .stFileUploader label {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Main container
#st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<h1 class="container">Hyperparameter Tuning Of Spam Email Classifier</h1>', unsafe_allow_html=True)

# Split the page into two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown('<h1 class="title">📧 Spam Email Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="subheader">🔍 Predict whether an email is spam or not</h2>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("📁 Choose a text or PDF file", type=["txt", "pdf"])

    # Preprocess text function
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\d', '', text)
        text = text.lower()
        text = text.split()
        text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
        text = ' '.join(text)
        return text

    # Extract text from PDF function
    def extract_text_from_pdf(file_path):
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    # Prediction
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            email_text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())
            email_text = extract_text_from_pdf("temp.pdf")
            os.remove("temp.pdf")  # Clean up the temporary file
        
        st.text_area('📄 Email content:', email_text, height=200)
        
        if st.button('📩 Scan'):
            processed_text = preprocess_text(email_text)
            text_vector = vectorizer.transform([processed_text]).toarray()
            prediction = model.predict(text_vector)
            
            if prediction[0] == 1:
                st.write('### The email is Scanned to be: **📬 Spam**')
            else:
                st.write('### The email is Scanned to be: **📧 Not Spam**')

with col2:
    st.image('C:/Users/PMLS/Downloads/Spam Email Scanner/2.jpg', caption='', use_column_width=True)

# Footer
st.markdown('<div class="footer">Developed by Talal Hassan Khan</div>', unsafe_allow_html=True)

# Close main container
st.markdown('</div>', unsafe_allow_html=True)
