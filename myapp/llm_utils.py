# llm_utils.py
from collections import defaultdict
import os
import numpy as np
import pickle
import faiss
from openai import AzureOpenAI
from dotenv import load_dotenv
import re
import json
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

history = []

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
        results = search(prompt, top_k=5)
        context = ""
        for i, res in enumerate(results):
            print(f"[{i+1}] Titel: {res['metadata']['title']} ")
            context += f"Document {i+1}: {res['text']}...\n\n"

        enhanced_prompt = f"""
Please answer the following medical question based on the provided information.

You MUST include references to the specific documents you use.

        {context}

        Question: {prompt}

        Please provide a concise answer. Make references in the following format [number] after each statement based on the source. No referencelist should be included.
        """#Here is a history of previous asked and answered questions (can be empty if there is no history). Use only if history is deemd relevant to the question being asked. {history}"""

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
            temperature=0.2,
            top_p=0.5,
            model=GPT4O_DEPLOYMENT_NAME
        )
        the_llm_responce, ref = create_refrencelist(response.choices[0].message.content, results)
        print(response.choices[0].message.content)

        history.append({
            "question" : prompt,
            "response" :the_llm_responce,
            "references": ref
        })
        #Try to minimize amount of history, this is not yet used.
        if len(history) >= 3:
            history.remove(history[0])
        s = {
                "references": ref,
                "content": the_llm_responce,
            }
        return s
    except Exception as e:
        print(f"Error calling GPT-4o: {str(e)}")
        return f"Error generating response: {str(e)}"

def create_refrencelist(response, results):
    # mapping the number from the chunk to the url & find all ref. in the LLM response
    number_to_url = {i + 1: doc['metadata']['url'] for i, doc in enumerate(results)}
    matches = list(re.finditer(r'\[(\d+)\]', response))

    # get all the num,urls into a list
    ref_sequence = []
    for match in matches:
        num = int(match.group(1))
        url = number_to_url[num]
        ref_sequence.append(url)

    url_to_newnum = {}
    references = []
    updated_response = response
    current_number = 1


    # Want it to start at 1 always, as that is the convention and better for the user experiance
    for i, url in enumerate(ref_sequence):
        if url not in url_to_newnum:
            url_to_newnum[url] = current_number
            current_number += 1

    result_parts = []
    last_idx = 0
    for match, url in zip(matches, ref_sequence):
        start, end = match.span()
        result_parts.append(response[last_idx:start])
        result_parts.append(f"[{url_to_newnum[url]}]")
        last_idx = end
    result_parts.append(response[last_idx:])
    updated_response = ''.join(result_parts)

    #Remove the duplicates created when converting all chunk ids to doc ids [2][2] - > [2]
    updated_response = re.sub(r'\[(\d+)\]\[\1\]', r'[\1]', updated_response)

        #The ref. list is updated in accordance to the ref in the text, only the ones used get presented here
    references = []
    for url, newnum in sorted(url_to_newnum.items(), key=lambda x: x[1]):
        doc = next(doc for doc in results if doc['metadata']['url'] == url)
        references.append(
            f"[{newnum}] Title: {doc['metadata']['title']}\n"
            f"URL: {doc['metadata']['url']}\n\n"
        )

    return updated_response, ''.join(references)