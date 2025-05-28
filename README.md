# Resume Parser AI

An AI-powered resume parsing system that extracts and analyzes information from resumes using machine learning and natural language processing techniques.

## Features

- Extract key information from PDF resumes
- Parse and structure resume data
- Natural language processing for text analysis
- Machine learning-based information extraction
- Support for multiple resume formats

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ResumeParserAI.git
cd ResumeParserAI
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK resources:
```bash
python download_resources.py
```

5. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. The application will be available at `http://localhost:5000`

3. Upload a resume PDF file through the web interface to analyze it

## Project Structure

- `app.py` - Main Flask application
- `model.py` - Machine learning model implementation
- `dataset.py` - Dataset handling and preprocessing
- `download_resources.py` - Script to download required NLTK resources
- `test_live.py` - Testing script for live resume parsing

## Requirements

See `requirements.txt` for a complete list of dependencies.

## License

[Add your license information here]
