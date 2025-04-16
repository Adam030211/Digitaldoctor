import pandas as pd
import requests
import os
import fitz  # PyMuPDF for PDF processing
import re
import json
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

# Function to download a PDF from a URL
def download_pdf(url, output_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

# Function to process text (remove stopwords)
def process_text(text, swedish_stopwords):
    # Convert to lowercase
    text = text.lower()
    # Replace special characters with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove Swedish stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in swedish_stopwords]
    
    return ' '.join(filtered_words)

# Main function
def main():
    # Create necessary directories
    os.makedirs('pdfs', exist_ok=True)
    os.makedirs('processed_texts', exist_ok=True)
    
    # Load CSV file
    df = pd.read_csv("/Users/adampersson/Desktop/demo/demo/myapp/medical_guidelines.csv")
    
    # Define Swedish stopwords
    try:
        nltk.download('stopwords')
        swedish_stopwords = set(stopwords.words('swedish'))
    except:
        # If NLTK's Swedish stopwords aren't available, use a basic set
        swedish_stopwords = {'och', 'som', 'det', 'är', 'att', 'i', 'en', 'jag', 'hon', 'han', 
                           'den', 'för', 'med', 'på', 'till', 'av', 'om', 'så', 'men', 'de', 
                           'inte', 'har', 'du', 'kan', 'ett', 'vi', 'från', 'ska', 'måste', 'vara'}
    
    # Add additional Swedish stopwords
    additional_stopwords = {'eller', 'samt', 'också', 'även', 'då', 'när', 'där', 'hur', 'vad', 
                           'vilken', 'vem', 'vilket', 'detta', 'dessa', 'denna', 'detta', 'här', 
                           'där', 'man', 'sig', 'över', 'under', 'genom', 'efter', 'före', 'mellan'}
    swedish_stopwords.update(additional_stopwords)
    
    metadata = []
    
    # Process each PDF
    for index, row in tqdm(df.iterrows(), total=len(df)):
        title = row['title'].strip('"') if isinstance(row['title'], str) else "Untitled"
        url = row['url']
        category = row['category']
        
        # Create filename
        safe_title = re.sub(r'[^\w]', '_', title)
        pdf_filename = f"pdfs/{safe_title}.pdf"
        txt_filename = f"processed_texts/{safe_title}.txt"
        
        # Download PDF
        if download_pdf(url, pdf_filename):
            # Extract text
            text = extract_text_from_pdf(pdf_filename)
            
            if text:
                # Process text
                processed_text = process_text(text, swedish_stopwords)
                
                # Save processed text
                with open(txt_filename, 'w', encoding='utf-8') as f:
                    f.write(processed_text)
                
                # Add to metadata
                metadata.append({
                    'title': title,
                    'url': url,
                    'category': category,
                    'local_pdf_path': pdf_filename,
                    'processed_text_path': txt_filename
                })
    
    # Save metadata
    with open('metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()