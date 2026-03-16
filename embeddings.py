import streamlit as st
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    """Loads and caches the SentenceTransformer model."""
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise

def embed_texts(texts: list[str]) -> np.ndarray:
    """Generates embeddings for a list of texts."""
    try:
        model = load_embedding_model()
        embeddings = model.encode(texts)
        return embeddings
    except Exception as e:
        logger.error(f"Failed to embed texts: {e}")
        return np.array([])
