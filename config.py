import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_env_variable(var_name: str) -> str:
    """Retrieves an environment variable or raises an error if it's missing."""
    value = os.environ.get(var_name)
    if not value:
        raise ValueError(f"Missing required environment variable: {var_name}")
    return value

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GROQ_API_KEY = get_env_variable("GROQ_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
TAVILY_API_KEY = get_env_variable("TAVILY_API_KEY")
