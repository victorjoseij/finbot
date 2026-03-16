import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import logging
from models.llm import get_groq_client

logger = logging.getLogger(__name__)

@st.cache_data(show_spinner=False)
def generate_financial_analysis(transactions: List[Dict[str, Any]]) -> str:
    """Generates a comprehensive text analysis of the user's spending habits using Groq."""
    if not transactions:
        return "Not enough data for analysis."
        
    df = pd.DataFrame(transactions)
    total_spend = df["amount"].sum()
    cat_spend_str = df.groupby("category")["amount"].sum().to_string()
    
    top_txns = df.nlargest(5, "amount")[["description", "amount", "category"]].to_string(index=False)
    
    prompt = f"""
    You are an expert financial analyst. Please review the following aggregated bank transaction data and provide a concise, professional diagnostic assessment of the user's spending.
    
    Total Spending: {total_spend}
    Category Breakdown:
    {cat_spend_str}
    
    Top 5 Largest Transactions:
    {top_txns}
    
    Provide your analysis in 3 clear paragraphs:
    1. Overall Assessment (What does the overall spending say about their financial footprint?)
    2. Category Drill-down (Which categories dominate and why?)
    3. Actionable Recommendations (What should they look out for or optimize?)
    
    Ensure your response is plain text without markdown headers or asterisks, so it can easily be printed onto a PDF. Use simple bullet points (-) if needed. Do not use asterisks or hashes.
    """
    
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate analysis: {e}")
        return "We grouped and categorized your data, but real-time AI insight generation is temporarily unavailable."
