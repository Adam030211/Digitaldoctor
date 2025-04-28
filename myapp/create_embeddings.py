'''import os
import json
import numpy as np
import pickle
from openai import AzureOpenAI
from tqdm import tqdm

# Load environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_EMBEDDING_DEPLOYMENT_NAME = os.getenv("text-embedding-ada-002")  # Important: use an embedding model!

# Initialize Azure OpenAI Client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

# Set paths
METADATA_PATH = "metadata.json"
OUTPUT_EMBEDDINGS_PATH = "embeddings.npy"
OUTPUT_CHUNKS_PATH = "chunks.pkl"

# Embedding function
def get_embedding(text: str) -> list:
    response = client.embeddings.create(
        input=[text],
        model=AZURE_EMBEDDING_DEPLOYMENT_NAME
    )
    return response.data[0].embedding

# Load metadata
def load_metadata(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Chunk text
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# Create embeddings
def create_embeddings(metadata_path, output_embeddings_path, output_chunks_path):
    metadata = load_metadata(metadata_path)

    all_chunks = []
    chunk_metadata = []

    print(f"Processing {len(metadata)} documents...")

    for item in tqdm(metadata, desc="Loading documents"):
        try:
            with open(item["processed_text_path"], "r", encoding="utf-8") as f:
                text = f.read()
                chunks = chunk_text(text)

                for chunk in chunks:
                    all_chunks.append(chunk)
                    chunk_metadata.append({
                        "title": item["title"],
                        "url": item["url"],
                        "category": item["category"]
                    })

        except Exception as e:
            print(f"Error loading {item['processed_text_path']}: {e}")

    print(f"Total chunks to embed: {len(all_chunks)}")

    # Create embeddings
    embeddings = []
    for chunk in tqdm(all_chunks, desc="Creating embeddings"):
        embedding = get_embedding(chunk)
        embeddings.append(embedding)

    embeddings = np.array(embeddings).astype("float32")

    # Save embeddings and chunks
    np.save(output_embeddings_path, embeddings)
    with open(output_chunks_path, "wb") as f:
        pickle.dump({
            "chunks": all_chunks,
            "metadata": chunk_metadata
        }, f)

    print(f"Saved embeddings to {output_embeddings_path}")
    print(f"Saved chunk metadata to {output_chunks_path}")

if __name__ == "__main__":
    create_embeddings(
        metadata_path=METADATA_PATH,
        output_embeddings_path=OUTPUT_EMBEDDINGS_PATH,
        output_chunks_path=OUTPUT_CHUNKS_PATH
    )
'''