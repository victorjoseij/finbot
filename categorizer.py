import logging
import streamlit as st
from typing import List, Dict, Any
from models.llm import get_groq_client
from utils.confidence import compute_confidence

logger = logging.getLogger(__name__)

CATEGORIES = ["Food", "Shopping", "Bills & Utilities", "Travel", "Entertainment", "Health", "Subscriptions", "Miscellaneous"]

def categorize_with_groq(prompt: str) -> str:
    """Calls Groq LLM to categorize a single transaction."""
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return None

def categorize_single(transaction: Dict[str, Any], response_mode: str) -> Dict[str, Any]:
    """Categorizes a transaction by polling Groq LLM."""
    desc = transaction.get("description", "")
    amount = transaction.get("amount", "")
    
    prompt = f"Categorize this bank transaction into exactly one of these categories: {CATEGORIES}. Transaction: {desc} Amount: {amount}. Reply with ONLY the category name, nothing else."
    
    votes = []
    
    # Vote 1. Groq
    groq_vote = categorize_with_groq(prompt)
    if groq_vote and any(cat.lower() in groq_vote.lower() for cat in CATEGORIES):
        filtered_vote = next((cat for cat in CATEGORIES if cat.lower() in groq_vote.lower()), "Miscellaneous")
        votes.append(filtered_vote)
    elif groq_vote: 
        votes.append("Miscellaneous")
    
    if not votes:
        transaction.update({
            "category": "Unclassified",
            "confidence_score": 0.0,
            "confidence_label": "low",
            "model_votes": []
        })
        return transaction

    confidence = compute_confidence(votes)
    
    transaction.update({
        "category": confidence["category"],
        "confidence_score": confidence["score"],
        "confidence_label": confidence["label"],
        "model_votes": confidence["votes"]
    })
    
    return transaction
def categorize_all(transactions: List[Dict[str, Any]], response_mode: str) -> List[Dict[str, Any]]:
    """Categorizes all transactions sequentially with a progress bar."""
    enriched = []
    total = len(transactions)
    
    if total == 0:
        return enriched
        
    progress_bar = st.progress(0, text="Categorizing transactions...")
    
    for i, txn in enumerate(transactions):
        enriched_txn = categorize_single(txn, response_mode)
        enriched.append(enriched_txn)
        progress_bar.progress((i + 1) / total, text=f"Categorizing transactions... ({i+1}/{total})")
        
    progress_bar.empty()
    return enriched
