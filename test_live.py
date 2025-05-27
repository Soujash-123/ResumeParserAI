# test_live.py

import argparse
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
