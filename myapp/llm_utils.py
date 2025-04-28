#llm.utils.py
import json
import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
import requests
from openai import AzureOpenAI
from dotenv import load_dotenv
import re
#from unstructured.partition.pdf import partition_pdf #used earlier but overkill? (Det var för tufft för min data lokalt)
import pdfplumber
import nltk
from nltk.corpus import stopwords


#from pdfplumber import open as pdf_open

# Load the enviroment variables. Key and endpoint
load_dotenv()

# Azure OpenAI settings
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_DEPLOYMENT_NAME = "gpt-4o-mini"

swedish_stopwords = {'och', 'som', 'det', 'är', 'att', 'i', 'en', 'jag', 'hon', 'han', 
                           'den', 'för', 'med', 'på', 'till', 'av', 'om', 'så', 'men', 'de', 
                           'inte', 'har', 'du', 'kan', 'ett', 'vi', 'från', 'ska', 'måste', 'vara','patient','vård','sjuksköterska','läkare', 'doktor', 'avdelning'}


dataIsPreProcessed = False # Datan är redan preprocessed. Sätt till False om du vill uppdatera data

    
# Load and prepare metadata
def load_metadata():
    try:
        nltk.download('stopwords')
        swedish_stopwords = set(stopwords.words('swedish'))
    except:
        
        # If NLTK's Swedish stopwords aren't available, use a basic set
        swedish_stopwords = {'och', 'som', 'det', 'är', 'att', 'i', 'en', 'jag', 'hon', 'han', 
                           'den', 'för', 'med', 'på', 'till', 'av', 'om', 'så', 'men', 'de', 
                           'inte', 'har', 'du', 'kan', 'ett', 'vi', 'från', 'ska', 'måste', 'vara','patient','vård','sjuksköterska','läkare', 'doktor', 'avdelning'}
    
    # Add additional Swedish stopwords
    additional_stopwords = {'eller', 'samt', 'också', 'även', 'då', 'när', 'där', 'hur', 'vad', 
                           'vilken', 'vem', 'vilket', 'detta', 'dessa', 'denna', 'detta', 'här', 
                           'där', 'man', 'sig', 'över', 'under', 'genom', 'efter', 'före', 'mellan','patient','vård','sjuksköterska','läkare', 'doktor', 'avdelning','individ','obs','bör', '\n'}
    swedish_stopwords.update(additional_stopwords)

    with open('metadata.json', 'r', encoding='utf-8') as f:
        return json.load(f)
    
    
def extract_text_and_tables(filepath, output_dir="rag_chunks"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chunks = []

    try:
        with pdfplumber.open(filepath) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                # Extract all the text parts
                text = page.extract_text()
                if text:
                    print("Kommer till refine")
                    text = refine_query(text)
                    chunks.append({
                        "type": "text",
                        "page": page_number,
                        "content": text.strip()
                    })

                # Extarct all tables
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if not table:
                        continue  # hoppa tomma tabeller

                    df = pd.DataFrame(table)

                    # Text representation of the table to the
                    table_text = df.to_string(index=False, header=False)

                    table_text = refine_query(table_text)

                    chunks.append({
                        "type": "table",
                        "page": page_number,
                        "table_number": table_idx + 1,
                        "content": table_text.strip()
                    })
    except Exception as e:
        print(f"Error loading {filepath}: {e}")

    # Save the text and tables to a JSON file.
    filename = os.path.basename(filepath).replace('.pdf', '')
    output_path = os.path.join(output_dir, f"{filename}.json")
    #Added force_ascii = False so that we can handle the swedish letters
    pd.DataFrame(chunks).to_json(output_path, orient="records", indent=2, force_ascii=False) 
    return chunks

def preprocess_pdf(data):
    os.makedirs("rag_chunks", exist_ok=True)
    dataIsPreProcessed = True
    
    print(f"Before loop: {len(data)} items") # Detta är för debug. Ta bort sen...
    for item in data:
        try:
            extract_text_and_tables(item['local_pdf_path'],"rag_chunks")
        except Exception as e:
            print(f"Error extract_text_and_tables: {e}")

    """""
        print("In the loop")
        try:
            elements = partition_pdf(
                filename=item['local_pdf_path'],
                strategy="fast",  
                infer_table_structure=False
            )
            if not elements:  # Check for silent failures
                raise ValueError("No elements extracted (possibly too large)")
        except MemoryError:
            print(f"Skipping {item['local_pdf_path']} (memory error)")
            continue  # Skip to next file if the current can't be found
        except Exception as e:
            print(f"Error processing {item['local_pdf_path']}: {e}")
            traceback.print_exc()
            continue
        

        elements = smart_partition_pdf(item['local_pdf_path'])
        processed_data = []

        for elem in elements:

            if elem.category == "Table":
                print("A table")
                # Convert table to readable format
                df = pd.DataFrame(elem.metadata.text_as_html)
                table_desc = f"Table showing: {', '.join(df.columns)}\n Rows:\n{df.head(2).to_markdown()}"
                processed_data.append({
                    "text": table_desc,
                    "type": "table"
                })
            else:
                print("text")
                processed_data.append({
                    "text": str(elem),
                    "type": "text"
                    #"metadata": elem.metadata.to_dict() Might make to many tokens?
                })
        try:
            base_name = os.path.basename(item['local_pdf_path']) 
            output_path = os.path.join("preprocess_texts", f"{base_name}.jsonl")

            

            with open(output_path, 'w', encoding='utf-8') as f:
                print("Wrote a file")
                for data_item in processed_data:
                    f.write(json.dumps(data_item) + "\n") 
        except Exception as e:
            print(f"Failed to write {output_path}: {str(e)}")
        
        del elements 
        """
    return dataIsPreProcessed

    

# Load all processed texts
def load_processed_texts(metadata):
    documents = []
    for item in metadata:
        try:
            filename = os.path.basename(item['local_pdf_path']).replace('.pdf', '')
            input_path = os.path.join("rag_chunks", f"{filename}.json")
            with open(input_path, 'r', encoding='utf-8') as f:
                text = json.load(f)
                documents.append({
                    'text': text,
                    'title': item['title'],
                    'url': item['url'],
                    'category': item['category']
                })
            
        except Exception as e:
            print(f"Error loading {item['processed_text_path']}: {e}")
    return documents
"""""
def load_preprocessed_texts(metadata):
    documents = []
    for item in metadata:
        try:
            with open(item['processed_text_path'], 'r', encoding='utf-8') as f:
                text = f.read()
                documents.append({
                    'text': text,
                    'title': item['title'],
                    'url': item['url'],
                    'category': item['category']
                })
        except Exception as e:
            print(f"Error loading {item['processed_text_path']}: {e}")
    return documents

import traceback
"""
"""""
def smart_partition_pdf(filepath):
    all_elements = []
    total_pages = None

    # Försöker läsa antalet sidor i PDF först
    try:
        reader = PdfReader(filepath)
        total_pages = len(reader.pages)
        print(f"PDF {filepath} has {total_pages} pages.")
    except Exception as e:
        print(f"Could not read page count for {filepath}: {e}")
        return []

    for page_num in range(1, total_pages + 1):
        print(f"Processing page {page_num} of {total_pages}")
        try:
            elements = partition_pdf(
                filename=filepath,
                strategy="hi_res",
                infer_table_structure=True,
                pages=[page_num],
            )
            print(f"Successfully processed page {page_num} with hi_res.")
        except Exception as e:
            print(f"hi_res failed on page {page_num}: {e}")
            traceback.print_exc()

            try:
                elements = partition_pdf(
                    filename=filepath,
                    strategy="fast",
                    infer_table_structure=False,
                    pages=[page_num],
                )
                print(f"Fallback succeeded on page {page_num} with fast strategy.")
            except Exception as fallback_e:
                print(f"Fallback also failed on page {page_num}: {fallback_e}")
                traceback.print_exc()
                continue  # Gå vidare till nästa sida

        all_elements.extend(elements)

    return all_elements
"""


def create_embeddings_index_from_json(documents, chunk_size=512, overlap=50):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create chunks and metadata
    chunks = []
    chunk_metadata = []
    
    for doc in documents:
        # Make sure the necessary keys exist
        if 'text' in doc and 'title' in doc and 'url' in doc and 'category' in doc:

            try:
                text_data = doc['text']
                for element in text_data:
                    text = element["content"]
                    is_table = element["type"] == "table"
                    if text.strip():
                        words = text.split()
                        # Split text into chunks
                        for i in range(0, len(words), chunk_size - overlap):
                            chunk = ' '.join(words[i:i+chunk_size])
                            if len(chunk.strip()) > 0:
                                chunks.append(chunk)
                                chunk_metadata.append({
                                    'title': doc['title'],
                                    'url': doc['url'],
                                    'category': doc['category'],
                                    'chunk_id': len(chunks) - 1,
                                    'is_table': is_table  # Added that this is a table -> trying to make it better then previous RAG attempt
                                })
            except Exception as e:
                print(f"Error processing document {doc['title']}: {e}")
        else:
            print(f"Skipping document with missing fields: {doc}")
    
    # Create embeddings
    embeddings = model.encode(chunks)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return index, chunks, chunk_metadata, model

# Create embeddings and index
def create_embeddings_index(documents, chunk_size=512, overlap=50):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use a different model
    
    # Create chunks
    chunks = []
    chunk_metadata = []
    
    for doc in documents:
        text = doc['text']
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i+chunk_size])
            if len(chunk.strip()) > 0:
                chunks.append(chunk)
                chunk_metadata.append({
                    'title': doc['title'],
                    'url': doc['url'],
                    'category': doc['category'],
                    'chunk_id': len(chunks) - 1
                })
    
    # Create embeddings
    embeddings = model.encode(chunks)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return index, chunks, chunk_metadata, model

def refine_query(query):

    # Remove some common words in my test, Ideally we should use a lib but I had problems installing the swedish and med. ones
    keywords = [word for word in query.lower().split() if word not in swedish_stopwords]
    
    new_query = " ".join(keywords)

    new_query = re.sub(r'[^\w\s-]', ' ', new_query) 
    new_query = re.sub(r'\s+', ' ', new_query).strip().lower()
    return new_query

def retrieve_documents(query, index, chunks, chunk_metadata, model, top_k=10):
    refinedQuery = refine_query(query)
    query_embedding = model.encode([refinedQuery])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
    
    results = []
    for idx in indices[0]:
        if idx >= 0 and idx < len(chunks):  # Ensure valid index
            results.append({
                'text': chunks[idx],
                'metadata': chunk_metadata[idx],
                'relevance_score': 1.0 - distances[0][list(indices[0]).index(idx)] / 100  # Normalize distance
            })
    
    return results

# Format context for LLM
def format_context(results):
    context = "Relevant information from medical guidelines:\n\n"
    
    for i, result in enumerate(results):
        context += f"Document {i+1}: {result['metadata']['title']}\n"
        context += f"Source: {result['metadata']['url']}\n"
        context += f"Excerpt: {result['text'][:1000]}...\n\n"
    
    return context

# Get Azure OpenAI response
# First API call: Ask the model to use the function
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
# Get LLM response with RAG
def get_llm_response(prompt, deployment_name=AZURE_DEPLOYMENT_NAME, use_rag=True):
    
    if not use_rag:
        # Call Azure OpenAI directly without RAG
        return get_azure_openai_response(prompt)
    
    # Load data for RAG
    metadata = load_metadata()
    if not dataIsPreProcessed: #This is set to true if needed in the beginning. It is here for the debugging
        preprocess_pdf(metadata)
    documents = load_processed_texts(metadata)
    index, chunks, chunk_metadata, embedding_model = create_embeddings_index_from_json(documents)
    
    # Retrieve relevant documents
    results = retrieve_documents(prompt, index, chunks, chunk_metadata, embedding_model)
    
    # Create context-enhanced prompt
    context = format_context(results)
    enhanced_prompt = f"""Svara på följande medecinska fråga baserat på given information.
    
{context}

Fråga: {prompt}

Ge ett sammanfattat svar och inkludera referenser till vilka specefika medecinska riktlijer som används. Svara på svenska.
"""
    
    # Get response from Azure OpenAI
    return get_azure_openai_response(enhanced_prompt)

# Example of usage
if __name__ == "__main__":
    # Example query
    query = "What are the guidelines for ADHD treatment in adults?"
    
    # Get response
    response = get_llm_response(query)
    print(response)

 #test   

def get_azure_openai_response(prompt):


    client = AzureOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    )
    try:
        response = client.chat.completions.create(
            messages=[

                {
                "role": "user",
                "content": prompt,
                }
            ],
            max_tokens=500,
            temperature=0.5,
            top_p=1.0,
            model=AZURE_DEPLOYMENT_NAME
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error calling Azure OpenAI: {str(e)}")
        return f"Error generating response: {str(e)}"
    
 