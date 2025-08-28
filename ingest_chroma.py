import os
import sys
import numpy as np
import pymupdf 
import chromadb
from langchain_ollama import OllamaEmbeddings
from embeddings import ChromaDBEmbeddingFunction
from config import (
    CHROMA_GLOBAL_COLLECTION_NAME, 
    CHROMA_HOST, 
    CHROMA_PORT
)


chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

embedding_function = ChromaDBEmbeddingFunction(OllamaEmbeddings(model="mxbai-embed-large"))

print(chroma_client.list_collections())
if CHROMA_GLOBAL_COLLECTION_NAME in chroma_client.list_collections():
    chroma_client.delete_collection(name=CHROMA_GLOBAL_COLLECTION_NAME)

collection = chroma_client.get_or_create_collection(
    name=CHROMA_GLOBAL_COLLECTION_NAME,
    metadata={"description": "A collection for RAG with Ollama - PDF Data"},
    embedding_function=embedding_function
)

print(f"Using ChromaDB Collection: {CHROMA_GLOBAL_COLLECTION_NAME}")

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    """
    doc = pymupdf.open(pdf_path)
    text_list = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            text_list.append(text)
    return text_list

def add_documents_to_collection(documents, pdf_filename):
    """
    Add extracted documents to ChromaDB collection.
    """
    print(f"Ingesting {len(documents)} pages from {pdf_filename}")
    doc_ids = [f"{pdf_filename}_doc_{i}" for i in range(len(documents))]
    collection.add(documents=documents, ids=doc_ids)

# Directory containing PDFs
pdf_directory = "knowledge_base"

# Check if the directory exists
if not os.path.exists(pdf_directory):
    print(f"Error: Directory '{pdf_directory}' not found!")
    exit(1)

# Process all PDFs in the directory
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

if not pdf_files:
    print("No PDFs found in 'knowledge_base'!")
else:
    print(f"Found {len(pdf_files)} PDF(s) for ingestion.")

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Extracting text from: {pdf_file}")
    documents = extract_text_from_pdf(pdf_path)

    if documents:
        add_documents_to_collection(documents, pdf_file)
    else:
        print(f"Warning: No text extracted from {pdf_file}.")

print("✅ Ingestion completed!")
