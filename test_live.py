# resume_test_live.py

import argparse
import re
import nltk
import spacy
import fitz  # PyMuPDF
import json
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Downloads (run once)
nltk.download("punkt")
nltk.download("stopwords")
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

# ------------------- Extract text from PDF -------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))  # sort by y then x
        page_text = "\n".join(block[4] for block in blocks)
        full_text += page_text + "\n"
    return full_text

# ------------------- Resume Information Extraction -------------------
def extract_resume_information(text):
    resume_text = text.lower().replace('\n', ' ').replace('\r', ' ')
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

    # ----------- Section Detection -----------
    section_patterns = {
        'projects': r'(projects|notable projects)[\s:\n]+',
        'education': r'(education|academic background)[\s:\n]+',
        'certifications': r'(certifications|certificates)[\s:\n]+',
        'experience': r'(experience|work history|professional experience)[\s:\n]+',
        'skills': r'(skills|technical skills|technologies|tools)[\s:\n]+',
    }

    for section, pattern in section_patterns.items():
        match = re.search(pattern, resume_text, re.IGNORECASE)
        if match:
            start = match.end()
            end = len(resume_text)
            next_matches = [
                re.search(pat, resume_text[start:], re.IGNORECASE)
                for sec, pat in section_patterns.items() if sec != section
            ]
            next_starts = [m.start() for m in next_matches if m]
            if next_starts:
                end = start + min(next_starts)
            section_text = resume_text[start:end].strip()
            extracted_info[section] = sent_tokenize(section_text)[:5]

    # ----------- Keywords Extraction -----------
    tech_keywords = ['python', 'java', 'sql', 'javascript', 'ml', 'ai', 'cloud',
                     'docker', 'kubernetes', 'aws', 'azure', 'html', 'css', 'pytorch',
                     'tensorflow', 'c++', 'linux']
    job_titles = ['engineer', 'developer', 'manager', 'analyst', 'scientist', 'consultant',
                  'architect', 'lead', 'intern', 'administrator']

    for sent in sent_tokenize(resume_text):
        for keyword in tech_keywords:
            if re.search(r'\b' + keyword + r'\b', sent):
                extracted_info['technologies'].append(keyword)

        for title in job_titles:
            if re.search(r'\b\w*\s*' + title + r'\b', sent):
                extracted_info['job_titles'].append(title)

    for ent in doc.ents:
        if ent.label_ == "ORG":
            extracted_info['companies'].append(ent.text)
        elif ent.label_ == "DATE":
            if re.search(r'\b(year|month|yr|mo)\b', ent.text):
                extracted_info['duration'].append(ent.text)

    # Deduplicate all list entries
    for key in extracted_info:
        if isinstance(extracted_info[key], list):
            extracted_info[key] = list(set(extracted_info[key]))

    return extracted_info

# ------------------- CLI Interface -------------------
def main():
    parser = argparse.ArgumentParser(description="Extract structured details from a resume PDF.")
    parser.add_argument("pdf_path", help="Path to the resume PDF file")
    parser.add_argument("--pretty", action="store_true", help="Pretty print the JSON output")
    args = parser.parse_args()

    if not args.pdf_path.lower().endswith(".pdf"):
        print("❌ Please provide a valid PDF file.")
        return

    try:
        text = extract_text_from_pdf(args.pdf_path)
        info = extract_resume_information(text)

        # Output as JSON
        if args.pretty:
            print(json.dumps(info, indent=2))
        else:
            print(json.dumps(info))

    except Exception as e:
        print("❌ Error processing the PDF:", str(e))

if __name__ == "__main__":
    main()
