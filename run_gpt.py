import os
import sys
import numpy as np
import pymupdf 
import chromadb
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from embeddings import ChromaDBEmbeddingFunction
from config import (
    LLM_MODEL, 
    CHROMA_HOST, 
    CHROMA_PORT, 
    EMBEDDING_MODEL, 
    CHROMA_GLOBAL_COLLECTION_NAME, 
    CUSTOM_PROMPT
)

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

embedding_function = ChromaDBEmbeddingFunction(
    OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )
)
collection = chroma_client.get_or_create_collection(
    name=CHROMA_GLOBAL_COLLECTION_NAME,
    embedding_function=embedding_function
)

def query_chromadb(query_text, n_results=2):
    #print("Collecting docs from the chromaDB")
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results["documents"], results["metadatas"]

def query_ollama(prompt):
    #print("Running the LLM model", LLM_MODEL)
    llm = OllamaLLM(model=LLM_MODEL)
    return llm.invoke(prompt)

def rag_pipeline(query_text):
    retrieved_docs, metadata = query_chromadb(query_text)
    #print(retrieved_docs)
    context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."
    #print(CUSTOM_PROMPT)
    augmented_prompt = CUSTOM_PROMPT.format(context=context, query=query_text)
    response = query_ollama(augmented_prompt)
    return response

if len(sys.argv) < 2:
    print("Usage: python3 main.py \"<your query>\"")
    sys.exit(1)

query = sys.argv[1]
#print("Question: ", query)
response = rag_pipeline(query)
print(response)
