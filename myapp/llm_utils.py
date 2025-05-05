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
        create_refrencelist(response.choices[0].message.content, results)

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

    # FInd all the references that are formated [i], in accordance with the enhanced prompt
    ref_num_in_text = list(map(int, re.findall(r'\[(\d+)\]', response)))

        # Each number should be connected to said url, num-1 as we add one when sending it to the llm to match convention
    number_to_url = {num: results[num-1]['metadata']['url'] for num in ref_num_in_text}

    # Since we can get duplicates in the text, as the text might reference a chunk multiple times, we want to find all sources that ref. the same url (document)
    url_to_numbers = {}
    for num, url in number_to_url.items():
        if url not in url_to_numbers:
         url_to_numbers[url] = []
        url_to_numbers[url].append(num)

    #Multiple references can refrence the same dokument, just be from different chunks, 
    # this will give them the lowest value of the duplicates, to avoid multiple references that point to the same document, then the response text to the HTML page gets updated
    canonical_number = {}
    for url, nums in url_to_numbers.items():
        min_num = min(nums)
        for num in nums:
            canonical_number[num] = min_num

    updated_response = response
    for num in sorted(canonical_number, reverse=True): 
        updated_response = updated_response.replace(f"[{num}]", f"[{canonical_number[num]}]")

    #Create the ref list
    seen_urls = set()
    references = []
    ref_counter = 1  

    # Assign new reference numbers in order of first appearance as the conv. states to make it lättare
    for num in sorted(ref_num_in_text):
        url = number_to_url[num]
        if url not in seen_urls:
            doc = results[num-1]
            references.append(
                f"[{ref_counter}] Title: {doc['metadata']['title']}\n"
                f"URL: {doc['metadata']['url']}\n\n"
            )
            ref_counter += 1
            seen_urls.add(url)

    print(updated_response)
    #Find  example [2][2] in the text which steems from the chunk refs and turn them into [2] by finding patter that finds more then one [i] with the same number next to eachother
    double_ref_pattern = re.compile(r'\[(\d+)\]\[\1\]')

    double_ref_pattern.sub(r'[\1]', updated_response)

    return updated_response, "".join(references)

