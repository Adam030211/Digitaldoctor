# scraper.py
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
from .models import Document, DocumentChunk
url = "https://www.vgregion.se/halsa-och-vard/vardgivarwebben/vardriktlinjer/medicinska-och-vardadministrativa-riktlinjer/styrande-dokument-inom-halso--och-sjukvard/amnesomraden/alla-regionala-medicinska-riktlinjer/ "

def scrape_documents(url):
    """Scrape document information from the given URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        list_items = soup.find_all('li', class_='list-block__result-item')
        
        count = 0
        for item in list_items:
            link_element = item.find('a')
            if link_element:
                doc_url = link_element.get('href')
                title = link_element.get_text(strip=True)
                
                if doc_url and title:
                    # Check if document already exists
                    doc, created = Document.objects.get_or_create(
                        url=doc_url,
                        defaults={'title': title}
                    )
                    
                    if created:
                        count += 1
                        # Try to extract content if it's a PDF
                        if doc_url.lower().endswith('.pdf'):
                            try:
                                extract_pdf_content(doc)
                            except Exception as e:
                                print(f"Error extracting PDF content: {str(e)}")
        
        return count
    except Exception as e:
        print(f"Error scraping documents: {str(e)}")
        return 0

def extract_pdf_content(document):
    """Extract text from PDF and store in document"""
    try:
        response = requests.get(document.url)
        response.raise_for_status()
        
        # Open PDF file from bytes
        with io.BytesIO(response.content) as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            full_text = ""
            
            # Extract text from each page
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                full_text += page.extract_text() + "\n\n"
            
            # Save full text to document
            document.content = full_text
            document.save()
            
            # Create document chunks for better retrieval
            create_document_chunks(document)
            
    except Exception as e:
        print(f"Error in PDF extraction: {str(e)}")

def create_document_chunks(document, chunk_size=1000, overlap=100):
    """Break document into smaller chunks for better retrieval"""
    if not document.content:
        return
    
    # Delete existing chunks
    document.chunks.all().delete()
    
    text = document.content
    chunks = []
    
    # Simple chunking by character count with overlap
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]
        if chunk_text:
            chunks.append(chunk_text)
    
    # Save chunks to database
    for i, chunk_text in enumerate(chunks):
        DocumentChunk.objects.create(
            document=document,
            content=chunk_text,
            chunk_index=i
        )