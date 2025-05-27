#!/usr/bin/env python3
"""
Download script for NLTK and spaCy resources required by ResumeParserAI
Run this script before starting the application to ensure all resources are available.
"""

import nltk
import spacy
import sys
import os

def download_resources():
    print("Downloading NLTK resources...")
    resources = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'omw-1.4'
    ]
    
    for resource in resources:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource)
            print(f"✅ Successfully downloaded {resource}")
        except Exception as e:
            print(f"❌ Error downloading {resource}: {e}")
    
    print("\nDownloading spaCy models...")
    try:
        if not spacy.util.is_package("en_core_web_sm"):
            print("Downloading en_core_web_sm...")
            spacy.cli.download("en_core_web_sm")
            print("✅ Successfully downloaded en_core_web_sm")
        else:
            print("✅ en_core_web_sm is already installed")
    except Exception as e:
        print(f"❌ Error downloading spaCy model: {e}")
    
    print("\nResource download complete!")

if __name__ == "__main__":
    # Create NLTK data directory if it doesn't exist
    nltk_data_dir = os.path.expanduser("~/nltk_data")
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
        print(f"Created NLTK data directory at {nltk_data_dir}")
    
    download_resources()
    print("\nYou can now run the ResumeParserAI application.")
