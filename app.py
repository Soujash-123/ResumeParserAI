import joblib
import os
import re
import spacy
import nltk
import torch
import fitz  # PyMuPDF
import json
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import torch.nn as nn

# Download resources (first time only)
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    if not spacy.util.is_package("en_core_web_sm"):
        spacy.cli.download("en_core_web_sm")
except Exception as e:
    print(f"Warning: Error downloading resources: {e}")
    # Continue anyway as resources might be available
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

app = Flask(__name__)

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
model = joblib.load(os.path.join(MODEL_DIR, "resume_classifier_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

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
        doc = nlp(text.lower())
        skills = []
        tech_keywords = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin'],
            'frameworks': ['django', 'flask', 'spring', 'react', 'angular', 'vue', 'node.js', 'express'],
            'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle'],
            'cloud': ['aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes'],
            'tools': ['git', 'jenkins', 'jira', 'confluence', 'agile', 'scrum'],
            'ai_ml': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn']
        }
        for category, keywords in tech_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    skills.append(keyword)
        return list(set(skills))

    def calculate_match_score(self, resume_text, job_description):
        cleaned_resume = self.preprocessor.preprocess_text(resume_text)
        cleaned_job_desc = self.preprocessor.preprocess_text(job_description)
        tfidf_matrix = self.vectorizer.fit_transform([cleaned_resume, cleaned_job_desc])
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        resume_skills = set(self.extract_key_skills(cleaned_resume))
        job_skills = set(self.extract_key_skills(cleaned_job_desc))
        skill_match = len(resume_skills.intersection(job_skills)) / len(job_skills) if job_skills else 0

        final_score = (similarity_score * 0.7) + (skill_match * 0.3)
        return {
            'overall_match_score': round(final_score * 100, 2),
            'text_similarity_score': round(similarity_score * 100, 2),
            'skill_match_score': round(skill_match * 100, 2),
            'matching_skills': list(resume_skills.intersection(job_skills)),
            'missing_skills': list(job_skills - resume_skills)
        }

# ------------------- Extract text from PDF -------------------

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
        page_text = "\n".join(block[4] for block in blocks)
        full_text += page_text + "\n"
    return full_text

# ------------------- Predict Resume Category -------------------

def predict_resume_category(resume_text):
    preprocessor = Preprocessing()
    cleaned_text = preprocessor.preprocess_text(resume_text)
    transformed_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(transformed_text)
    return label_encoder.inverse_transform(prediction)[0]

# ------------------- Resume Information Extraction -------------------
def extract_resume_information(text):
    resume_text = text.replace('\r', '').replace('\n', '\n')  # keep newlines
    extracted_info = {
        'name': None,
        'education': [],
        'experience': [],
        'skills': [],
        'projects': [],
        'certifications': [],
        'companies': [],
        'job_titles': [],
        'technologies': [],
        'duration': []
    }

    doc = nlp(text)

    # ----------- Name Extraction -----------
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.split()) <= 4:
            extracted_info['name'] = ent.text.strip()
            break

    # ----------- Section-wise Line Grouping -----------
    SECTION_MAP = {
        'education': ['education', 'academic background', 'studies'],
        'experience': ['experience', 'work history', 'professional background'],
        'skills': ['skills', 'technical skills', 'technologies', 'tools'],
        'projects': ['projects', 'notable projects', 'personal projects'],
        'certifications': ['certifications', 'certificates']
    }

    def normalize(text):
        return text.strip().lower()

    def detect_section_header(line):
        line = normalize(line)
        for section, keywords in SECTION_MAP.items():
            for keyword in keywords:
                if keyword in line and len(line) < 60:  # likely a section title
                    return section
        return None

    lines = text.splitlines()
    current_section = None

    for line in lines:
        header = detect_section_header(line)
        if header:
            current_section = header
            continue

        if current_section and line.strip():
            extracted_info[current_section].append(line.strip())

    # Limit each section to top 5 sentences
    for key in SECTION_MAP.keys():
        if isinstance(extracted_info[key], list):
            extracted_info[key] = sent_tokenize(" ".join(extracted_info[key]))[:5]

    # ----------- Keywords Extraction -----------
    tech_keywords = ['python', 'java', 'sql', 'javascript', 'ml', 'ai', 'cloud',
                     'docker', 'kubernetes', 'aws', 'azure', 'html', 'css', 'pytorch',
                     'tensorflow', 'c++', 'linux']
    job_titles = ['engineer', 'developer', 'manager', 'analyst', 'scientist', 'consultant',
                  'architect', 'lead', 'intern', 'administrator']

    for sent in sent_tokenize(resume_text.lower()):
        for keyword in tech_keywords:
            if keyword in sent:
                extracted_info['technologies'].append(keyword)
        for title in job_titles:
            if title in sent:
                extracted_info['job_titles'].append(title)

    for ent in doc.ents:
        if ent.label_ == "ORG":
            extracted_info['companies'].append(ent.text)
        elif ent.label_ == "DATE":
            extracted_info['duration'].append(ent.text)

    # Deduplicate all list entries
    for key in extracted_info:
        if isinstance(extracted_info[key], list):
            extracted_info[key] = list(set(extracted_info[key]))

    return extracted_info

# ------------------- Flask Routes -------------------

@app.route('/parse-resume', methods=['POST'])
def parse_resume():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Please provide a PDF file'}), 400

    try:
        # Save the file temporarily
        temp_path = 'temp_resume.pdf'
        file.save(temp_path)
        
        # Extract text and information
        text = extract_text_from_pdf(temp_path)
        info = extract_resume_information(text)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process-resume', methods=['POST'])
def process_resume():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Please provide a PDF file'}), 400

    try:
        # Save the file temporarily
        temp_path = 'temp_resume.pdf'
        file.save(temp_path)
        
        # Extract text
        resume_text = extract_text_from_pdf(temp_path)
        
        # Predict category
        category = predict_resume_category(resume_text)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({
            'predicted_category': category,
            'resume_text': resume_text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/match-resume', methods=['POST'])
def match_resume():
    if 'file' not in request.files or 'job_description' not in request.form:
        return jsonify({'error': 'Both resume file and job description are required'}), 400
    
    file = request.files['file']
    job_description = request.form['job_description']
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Please provide a PDF file'}), 400

    try:
        # Save the file temporarily
        temp_path = 'temp_resume.pdf'
        file.save(temp_path)
        
        # Extract text
        resume_text = extract_text_from_pdf(temp_path)
        
        # Match with job description
        matcher = ResumeMatcher()
        match_results = matcher.calculate_match_score(resume_text, job_description)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify(match_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """Home page for the Resume Parser AI application"""
    return render_template('index.html')

@app.route('/docs')
def docs():
    """Documentation page for the Resume Parser AI API"""
    return render_template('documentation.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
