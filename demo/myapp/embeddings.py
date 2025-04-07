# embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
from .models import DocumentChunk

# Load model once
model = None

def get_embedding_model():
    global model
    if model is None:
        # Choose a smaller model for efficiency
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def create_embeddings_for_chunks():
    """Create embeddings for all chunks without embeddings"""
    chunks = DocumentChunk.objects.filter(embedding__isnull=True)
    if not chunks:
        return 0
    
    model = get_embedding_model()
    
    # Process in batches to avoid memory issues
    batch_size = 32
    count = 0
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [chunk.content for chunk in batch]
        embeddings = model.encode(texts)
        
        # Save embeddings
        for j, chunk in enumerate(batch):
            chunk.embedding = embeddings[j].tolist()
            chunk.save()
            count += 1
    
    return count

def search_similar_chunks(query, top_k=5):
    """Find chunks similar to query"""
    model = get_embedding_model()
    query_embedding = model.encode(query)
    
    # Get all chunks with embeddings
    chunks = DocumentChunk.objects.filter(embedding__isnull=False)
    
    # Calculate similarity scores
    results = []
    for chunk in chunks:
        chunk_embedding = np.array(chunk.embedding)
        similarity = np.dot(query_embedding, chunk_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
        )
        results.append((chunk, similarity))
    
    # Sort by similarity and return top_k
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]