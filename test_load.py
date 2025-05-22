# load_resume_model.py

import joblib
import os
import re
import spacy
import nltk
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# ------------------- Resume Preprocessor -------------------

class Preprocessing(nn.Module):
    def __init__(self):
        super(Preprocessing, self).__init__()

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s.]', '', text)
        text = re.sub(r'(?i)(?<=\b[a-z])\.(?=[a-z]{2,}\b)', '', text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r'\r\n', ' ', text)
        tokens = text.split()
        return ' '.join(tokens)

# ------------------- Load Model Components -------------------

MODEL_DIR = "saved_model"
model_path = os.path.join(MODEL_DIR, "resume_classifier_model.pkl")
vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

if not (os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(label_encoder_path)):
    raise FileNotFoundError("Model or related files not found. Please ensure 'saved_model/' contains all components.")

# Load saved components
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
label_encoder = joblib.load(label_encoder_path)

# ------------------- Resume Matching -------------------

class ResumeMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        self.preprocessor = Preprocessing()

    def extract_key_skills(self, text):
        """Extract key skills and technologies from text."""
        doc = nlp(text.lower())
        skills = []
        
        # Common technical skills and technologies
        tech_keywords = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin'],
            'frameworks': ['django', 'flask', 'spring', 'react', 'angular', 'vue', 'node.js', 'express'],
            'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle'],
            'cloud': ['aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes'],
            'tools': ['git', 'jenkins', 'jira', 'confluence', 'agile', 'scrum'],
            'ai_ml': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn']
        }
        
        # Extract skills based on keywords
        for category, keywords in tech_keywords.items():
            for keyword in keywords:
                if keyword in text.lower():
                    skills.append(keyword)
        
        return list(set(skills))

    def calculate_match_score(self, resume_text, job_description):
        """Calculate match score between resume and job description."""
        # Preprocess both texts
        cleaned_resume = self.preprocessor.preprocess_text(resume_text)
        cleaned_job_desc = self.preprocessor.preprocess_text(job_description)
        
        # Vectorize both texts
        tfidf_matrix = self.vectorizer.fit_transform([cleaned_resume, cleaned_job_desc])
        
        # Calculate cosine similarity
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Extract and compare skills
        resume_skills = set(self.extract_key_skills(cleaned_resume))
        job_skills = set(self.extract_key_skills(cleaned_job_desc))
        
        # Calculate skill match percentage
        if job_skills:
            skill_match = len(resume_skills.intersection(job_skills)) / len(job_skills)
        else:
            skill_match = 0
        
        # Combine scores (70% similarity, 30% skill match)
        final_score = (similarity_score * 0.7) + (skill_match * 0.3)
        
        return {
            'overall_match_score': round(final_score * 100, 2),
            'text_similarity_score': round(similarity_score * 100, 2),
            'skill_match_score': round(skill_match * 100, 2),
            'matching_skills': list(resume_skills.intersection(job_skills)),
            'missing_skills': list(job_skills - resume_skills)
        }

# ------------------- Use the Model -------------------

def predict_resume_category(resume_text):
    preprocessor = Preprocessing()
    cleaned_text = preprocessor.preprocess_text(resume_text)
    transformed_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(transformed_text)
    predicted_category = label_encoder.inverse_transform(prediction)[0]
    return predicted_category

# ------------------- Example Usage -------------------

if __name__ == "__main__":
    # Example input resume
    test_resume = """
    Experienced Software Engineer with 5+ years in Python, Java, and cloud platforms like AWS.
    Developed scalable backend systems and REST APIs. Familiar with ML models and DevOps tools.
    """
    
    # Example job description
    test_job_desc = """
    Looking for a Senior Software Engineer with strong Python and AWS experience.
    Must have experience with REST APIs and microservices architecture.
    Knowledge of machine learning and cloud technologies is a plus.
    """
    
    # Get category prediction
    category = predict_resume_category(test_resume)
    print("Predicted Resume Category:", category)
    
    # Get match score
    matcher = ResumeMatcher()
    match_results = matcher.calculate_match_score(test_resume, test_job_desc)
    print("\nResume Match Results:")
    print(f"Overall Match Score: {match_results['overall_match_score']}%")
    print(f"Text Similarity Score: {match_results['text_similarity_score']}%")
    print(f"Skill Match Score: {match_results['skill_match_score']}%")
    print("\nMatching Skills:", match_results['matching_skills'])
    print("Missing Skills:", match_results['missing_skills'])
