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

def update_valid_history(prompt):
    global history
    history_prompt = f"""Find the index values of all history instances: {history}, that are concidered the most relevant for the following question: {prompt}. Return in a list with the following format [index, index,..., index]. Do not return anything but the list """
    try:
        response = chat_client.chat.completions.create(
           messages=[
               {
                    "role": "user",
                    "content": history_prompt,

                }
            ],
            max_tokens=4096,
            temperature=0.2,
            top_p=0.5,
          model=GPT4O_DEPLOYMENT_NAME
       )
        newHistory = []
        indexes = []

        get_best_response = response.choices[0].message.content
        print("\n***History response***:" + get_best_response +"\n\n")
        indexes = list(re.findall(r'\d+', get_best_response))
  
        if len(indexes)> 0:
            indexes = [int(i) for i in indexes]
            indexes.sort()
            for i in indexes:
                if i < len(history):
                    newHistory.append(history[i])
            history=newHistory
        elif len(indexes) == 0:
            history = []
        #to ensure history doesn't become to big if the chat doesn't work we have to forcefully remove some
        elif len(history)>=5:
            history.pop(0)
        
    except Exception as e:
        print(f"Error calling GPT-4o for update_valid_history: {str(e)}")
    return history

# ---- LLM Response ----
def get_llm_response(prompt, use_rag=True):
    original_prompt = prompt
    global history

    if len(history)>=1:
        try:
            history = update_valid_history(prompt)
        
        except Exception as e:
            print(f"update_valid history does not work: {str(e)}")
    
    if len(history) >=1:
        try:
            prompt = f"{history}  {prompt} "
        except Exception as e:
            print(f"error constructing new prompt : {str(e)}")
    print(history)

    if use_rag:
        try:
            results = search(prompt, top_k=5)
        except Exception as e:
            print(f"error with search function: {str(e)}")
        print("Made it to creation of context")
        context = ""
        for i, res in enumerate(results):
            #print(f"[{i+1}] Titel: {res['metadata']['title']} ")
            context += f"Document {i+1}: {res['text']}...\n\n"

        enhanced_prompt = f"""
Please answer the following medical question based on the provided information.

You MUST include references to the specific documents you use.

        {context}

        Question: {original_prompt}

        Please provide a concise answer. Make references in the following format [number] after each statement based on the source. No referencelist should be included."""

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
        #print(response.choices[0].message.content)
        llm_responce_without_ref = remove_references(the_llm_responce)
 
        history.append(
            {"role": "user", "content": original_prompt}
        )
        history.append(
            {"role": "system", "content": llm_responce_without_ref}
        )
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

def remove_references(response):
    return re.sub(r'\[(\d+)\]',"", response)
