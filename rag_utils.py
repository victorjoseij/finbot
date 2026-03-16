import os
import faiss
import logging
import pdfplumber
import numpy as np
import streamlit as st
from typing import Tuple, List, Any
from models.embeddings import embed_texts

logger = logging.getLogger(__name__)

@st.cache_data
def build_vector_store(docs_path: str) -> Tuple[Any, List[str]]:
    """Loads PDF document, chunks text, and builds a FAISS index."""
    chunks = []
    chunk_size = 300
    overlap = 50
    
    try:
        if not os.path.exists(docs_path):
            logger.warning(f"Knowledge base document not found: {docs_path}")
            return None, []
            
        with pdfplumber.open(docs_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        
        start = 0
        while start < len(full_text):
            chunk = full_text[start:start+chunk_size]
            chunks.append(chunk)
            start += chunk_size - overlap
            
        if not chunks:
            return None, []
            
        embeddings = embed_texts(chunks)
        if embeddings.size == 0:
            return None, []
            
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        return index, chunks
        
    except Exception as e:
        logger.error(f"Error building vector store: {e}")
        return None, []

def retrieve_context(query: str, index: Any, chunks: List[str], top_k: int = 3) -> str:
    """Retrieves relevant text chunks for a query from the FAISS index."""
    if index is None or not chunks:
        return ""
        
    try:
        query_embedding = embed_texts([query]).astype('float32')
        if query_embedding.size == 0:
            return ""
            
        distances, indices = index.search(query_embedding, top_k)
        
        results = []
        for i in indices[0]:
            if 0 <= i < len(chunks):
                results.append(chunks[i])
                
        return "\n\n".join(results)
    except Exception as e:
        logger.error(f"Error retrieving RAG context: {e}")
        return ""
