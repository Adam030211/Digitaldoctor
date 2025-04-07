# llm_utils.py
import ollama
from .embeddings import search_similar_chunks
import traceback

def get_llm_response(prompt, model_name="llama3.2:latest", use_rag=True):
    """
    Get a response from the Ollama model with RAG if enabled
    
    Args:
        prompt (str): The input prompt for the model
        model_name (str): The name of the model as registered in Ollama
        use_rag (bool): Whether to use RAG
        
    Returns:
        str: The model's response
    """
    try:
        # If RAG is enabled, search for relevant document chunks
        context = ""
        if use_rag:
            similar_chunks = search_similar_chunks(prompt)
            if similar_chunks:
                context = "Here are some relevant documents to help answer the question:\n\n"
                for i, (chunk, score) in enumerate(similar_chunks):
                    doc_title = chunk.document.title
                    context += f"Document {i+1}: {doc_title}\n"
                    context += f"Content: {chunk.content[:500]}...\n\n"
                context += "Based on these documents, please answer the question.\n\n"
        
        # Create full prompt with context
        full_prompt = context + prompt if context else prompt
        
        response = ollama.chat(model=model_name, messages=[
            {
                'role': 'user',
                'content': full_prompt,
            }
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}\nFull traceback: {traceback.format_exc()}"