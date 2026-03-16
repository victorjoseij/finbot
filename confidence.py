from collections import Counter
from typing import List, Dict, Any

def compute_confidence(votes: List[str]) -> Dict[str, Any]:
    """Computes confidence score and label based on model agreement."""
    if not votes:
        return {"category": "Unclassified", "score": 0.0, "label": "low", "votes": []}
        
    vote_counts = Counter(votes)
    most_common_category, most_common_count = vote_counts.most_common(1)[0]
    
    score = most_common_count / len(votes)
    
    if score == 1.0:
        label = "high"
    elif score >= 0.66: 
        label = "medium"
    else:
        label = "low"

    return {
        "category": most_common_category,
        "score": score,
        "label": label,
        "votes": votes
    }
