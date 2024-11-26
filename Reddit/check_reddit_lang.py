import pandas as pd
from langdetect import detect
from collections import Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_language(text):
    try:
        # Convert to string and strip whitespace
        text = str(text).strip()
        
        # Check for invalid inputs
        if (pd.isna(text) or 
            text == '' or 
            len(text) < 3 or
            # Check if text contains any letters
            not any(c.isalpha() for c in text) or
            # Check if text is just numbers and special characters
            all(not c.isalpha() for c in text)):
            return 'unknown'
            
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Check if text is just a single word
        if len(text.split()) <= 1:
            return 'unknown'
            
        return detect(text)
    except Exception as e:
        logger.warning(f"Could not detect language: {str(e)}")
        return 'unknown'

def analyze_language_distribution(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Only analyze body columns (which is the top comments)
    columns_to_analyze = ['body']
    
    for column in columns_to_analyze:
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in CSV file")
            continue
            
        print(f"\nAnalyzing column: {column}")
        print("=" * 50)
        
        # Detect language for each text entry
        logger.info(f"Detecting languages for column: {column}")
        languages = [detect_language(text) for text in df[column]]
        
        # Count language frequencies
        lang_distribution = Counter(languages)
        
        # Calculate percentages
        total = len(languages)
        lang_percentages = {lang: (count/total)*100 for lang, count in lang_distribution.items()}
        
        # Print results
        print("\nLanguage Distribution:")
        print("-" * 30)
        for lang, percentage in sorted(lang_percentages.items(), key=lambda x: x[1], reverse=True):
            print(f"{lang}: {percentage:.2f}% ({lang_distribution[lang]} texts)")

if __name__ == "__main__":
    csv_path = "Reddit_2021_to_2024.csv"
    
    try:
        analyze_language_distribution(csv_path)
    except Exception as e:
        logger.error(f"Error analyzing language distribution: {str(e)}")
