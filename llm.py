import os
import logging
from config.config import GROQ_API_KEY
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_groq_client() -> Groq:
    """Initializes and returns the Groq client."""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        raise
