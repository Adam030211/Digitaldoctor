import json
import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

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

# Get LLM response with RAG
def get_llm_response(prompt, model_name="llama3.2:latest", use_rag=True):
    if not use_rag:
        response = ollama.chat(model=model_name, messages=[
            {"role": "user", "content": prompt}
        ])
        return response['message']['content']
    
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
    
    # Get response from LLM
    response = ollama.chat(model=model_name, messages=[
        {"role": "user", "content": enhanced_prompt}
    ])
    
    return response['message']['content']

# Example of usage
if __name__ == "__main__":
    # Example query
    query = "What are the guidelines for ADHD treatment in adults?"
    
    # Get response
    response = get_llm_response(query)
    print(response)