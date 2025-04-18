#llm.utils.py
import json
import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
import requests

# Azure OpenAI settings
AZURE_OPENAI_ENDPOINT = "https://exjobbaa3647630925.openai.azure.com/"
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = "2023-05-15"  # Update to the latest version if needed

# Load and prepare metadata
def load_metadata():
    with open('metadata.json', 'r', encoding='utf-8') as f:
        return json.load(f)

# Load all processed texts
def load_processed_texts(metadata):
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

# Retrieve relevant documents
def retrieve_documents(query, index, chunks, chunk_metadata, model, top_k=3):
    query_embedding = model.encode([query])
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
        context += f"Excerpt: {result['text'][:300]}...\n\n"
    
    return context

# Get Azure OpenAI response
def get_azure_openai_response(prompt, deployment_name="gpt-4"):
    """
    Get response from Azure OpenAI
    """
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }
    
    try:
        api_url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{deployment_name}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
        
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        response_data = response.json()
        
        return response_data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Azure OpenAI API Error: {e}")
        return f"Error generating response: {str(e)}"

# Get LLM response with RAG
def get_llm_response(prompt, deployment_name="gpt-4", use_rag=True):
    if not use_rag:
        # Call Azure OpenAI directly without RAG
        return get_azure_openai_response(prompt, deployment_name)
    
    # Load data for RAG
    metadata = load_metadata()
    documents = load_processed_texts(metadata)
    index, chunks, chunk_metadata, embedding_model = create_embeddings_index(documents)
    
    # Retrieve relevant documents
    results = retrieve_documents(prompt, index, chunks, chunk_metadata, embedding_model)
    
    # Create context-enhanced prompt
    context = format_context(results)
    enhanced_prompt = f"""Please answer the following medical question based on the provided information.
    
{context}

Question: {prompt}

Please provide a concise answer and include references to the specific medical guidelines you used.
"""
    
    # Get response from Azure OpenAI
    return get_azure_openai_response(enhanced_prompt, deployment_name)

# Example of usage
if __name__ == "__main__":
    # Example query
    query = "What are the guidelines for ADHD treatment in adults?"
    
    # Get response
    response = get_llm_response(query)
    print(response)