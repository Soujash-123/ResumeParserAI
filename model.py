# intelligent_resume_screening_local.py

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import spacy

from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import joblib

# ------------------- Download resources -------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# ------------------- Load Dataset -------------------
DATA_PATH = "./1/UpdatedResumeDataSet.csv"  # Make sure this CSV is in the same directory
df = pd.read_csv(DATA_PATH)

# ------------------- Preprocessing -------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
category = df["Category"].unique().tolist()

class Preprocessing(nn.Module):
    def __init__(self, cat_list):
        super(Preprocessing, self).__init__()
        self.catl = cat_list

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s.]', '', text)
        text = re.sub(r'(?i)(?<=\b[a-z])\.(?=[a-z]{2,}\b)', '', text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r'\r\n', ' ', text)
        tokens = text.split()
        return ' '.join(tokens)

    def forward(self, data):
        df = data.copy()
        df['Resume'] = df['Resume'].apply(self.preprocess_text)
        return df

preprocess = Preprocessing(category)
df_cleaned = preprocess(df)

# ------------------- WordCloud -------------------
all_sentences = ' '.join(df_cleaned["Resume"].values)
total_words = [
    token for sent in df_cleaned["Resume"].values for token in sent.split()
    if token not in stop_words and len(token) > 1 and not token.isdigit()
]

wordFreq = nltk.FreqDist(total_words)
wordcloud = WordCloud().generate(all_sentences)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# ------------------- Vectorization & Encoding -------------------
le = LabelEncoder()
vectorizer = TfidfVectorizer()
df_cleaned["enc_category"] = le.fit_transform(df_cleaned["Category"])
X = vectorizer.fit_transform(df_cleaned["Resume"])
y = df_cleaned["enc_category"]

# ------------------- Model Training -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
ovrc_model = OneVsRestClassifier(LogisticRegression())
ovrc_model.fit(X_train, y_train)
y_pred = ovrc_model.predict(X_test)

print("\n----- Classification Report -----")
print(classification_report(y_test, y_pred, target_names=category))

# ------------------- Sample Prediction -------------------
sample_resume = df["Resume"][69]
cleaned_sample = preprocess.preprocess_text(sample_resume)
sample_vector = vectorizer.transform([cleaned_sample])
pred = ovrc_model.predict(sample_vector)

print("\n----- Sample Prediction -----")
print("Predicted:", le.inverse_transform(pred)[0])
print("Actual:", df_cleaned["Category"][69])

# ------------------- Save Model and Tools -------------------
MODEL_DIR = "saved_model"
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(ovrc_model, os.path.join(MODEL_DIR, "resume_classifier_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("\n✅ Model, vectorizer, and label encoder saved in 'saved_model/' directory.")
