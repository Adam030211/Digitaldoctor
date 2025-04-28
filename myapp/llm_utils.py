# llm_utils.py
import os
import numpy as np
import pickle
import faiss
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---- Azure OpenAI Setup ----
# Chat (gpt-4o-mini)
GPT4O_API_KEY = os.getenv("GPT4O_API_KEY")
GPT4O_ENDPOINT = os.getenv("GPT4O_ENDPOINT")
GPT4O_DEPLOYMENT_NAME = "gpt-4o-mini" #os.getenv("gpt-4o-mini")

# Embedding
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT")
EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002" #os.getenv("EMBEDDING_DEPLOYMENT_NAME")

# Initialize separate Azure OpenAI clients
chat_client = AzureOpenAI(
    api_key=GPT4O_API_KEY,
    azure_endpoint=GPT4O_ENDPOINT,
    api_version="2024-12-01-preview"
)

embedding_client = AzureOpenAI(
    api_key=EMBEDDING_API_KEY,
    azure_endpoint=EMBEDDING_ENDPOINT,
    api_version="2024-12-01-preview"
)

# ---- Load FAISS index ----
embeddings = np.load("embeddings.npy")   # TODO: Correct the path
with open("chunks.pkl", "rb") as f:       # TODO: Correct the path
    data = pickle.load(f)

chunks = data["chunks"]
metadata = data["metadata"]

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ---- Search (RAG retrieval) ----
def search(query, top_k=3):
    response = embedding_client.embeddings.create(
        input=[query],
        model=EMBEDDING_DEPLOYMENT_NAME
    )
    query_embedding = np.array(response.data[0].embedding).astype('float32').reshape(1, -1)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append({
                "text": chunks[idx],
                "metadata": metadata[idx]
            })
    return results

# ---- LLM Response ----
def get_llm_response(prompt, use_rag=True):
    if use_rag:
        results = search(prompt, top_k=3)
        context = ""
        for i, res in enumerate(results):
            context += f"Document {i+1}: {res['metadata']['title']}\nSource: {res['metadata']['url']}\nExcerpt: {res['text'][:300]}...\n\n"

        enhanced_prompt = f"""
Please answer the following medical question based on the provided information.

You MUST include references to the specific documents you use.
Always include the TITLE and the SOURCE URL when citing.

        {context}

        Question: {prompt}

        Please provide a concise answer and include references.
        """
    else:
        enhanced_prompt = prompt

    try:
        response = chat_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": enhanced_prompt,
                }
            ],
            max_tokens=4096,
            temperature=1.0,
            top_p=1.0,
            model=GPT4O_DEPLOYMENT_NAME
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling GPT-4o: {str(e)}")
        return f"Error generating response: {str(e)}"

