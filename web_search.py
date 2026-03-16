import logging
from tavily import TavilyClient
from config.config import TAVILY_API_KEY

logger = logging.getLogger(__name__)

def tavily_search(query: str) -> str:
    """Performs a web search using Tavily API."""
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, search_depth="basic", max_results=3)
        
        results = response.get('results', [])
        if not results:
            return "No internet search results found."
            
        summary = "Web Search Results:\n"
        for i, res in enumerate(results):
            summary += f"{i+1}. {res.get('title', 'No Title')} - {res.get('content', '')[:200]}...\n"
            
        return summary
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return "Search unavailable."
