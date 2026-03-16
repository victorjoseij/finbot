import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from utils.parser import parse_csv, parse_pdf
from utils.categorizer import categorize_all
from utils.rag_utils import build_vector_store, retrieve_context
from utils.web_search import tavily_search
from models.llm import get_groq_client
from utils.report import generate_pdf_report
from utils.insight_generator import generate_financial_analysis

import tempfile
import os

# Must be the first Streamlit command
st.set_page_config(page_title="FinBot — Personal Finance Intelligence", layout="wide")

st.title("FinBot — Personal Finance Intelligence")

# --- UI Setup ---
st.sidebar.header("Settings")
llm_provider = st.sidebar.selectbox("LLM Provider", ["Groq"])
response_mode = st.sidebar.radio("Response Mode", ["Concise", "Detailed"])

st.sidebar.divider()
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Bank Statement (PDF or CSV)", type=["pdf", "csv"])

analyze_button = st.sidebar.button("Analyze Statement")

st.sidebar.divider()
st.sidebar.info("About\nFinBot extracts, categorizes, and lets you chat with your bank transactions using AI.")

# --- Initialization ---
if "transactions" not in st.session_state:
    st.session_state.transactions = []
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

try:
    index, chunks = build_vector_store("knowledge_base/finance_guide.pdf")
except Exception as e:
    st.warning("Could not build knowledge base vector store. RAG features will be limited.")
    index, chunks = None, []

# --- Main Layout ---
tab1, tab2, tab3 = st.tabs(["Statement Analysis", "Chat with your data", "Insights"])

def format_confidence(label: str) -> str:
    """Formats the confidence label into a colored badge sting."""
    if label == "high":
        return "🟢 [HIGH]"
    elif label == "medium":
        return "🟡 [MED]"
    else:
        return "🔴 [LOW]"

def get_chat_response(messages, provider):
    """Calls the selected provider for chat responses."""
    try:
        if provider == "Groq":
            client = get_groq_client()
            resp = client.chat.completions.create(messages=messages, model="llama-3.3-70b-versatile")
            return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling {provider}: {e}")
        return "Sorry, I encountered an error answering that."

# --- Data Processing Logic ---
def robust_parse_dates(date_series: pd.Series) -> pd.Series:
    """Robustly converts raw strings to datetime objects handling ambiguous DD MMM formats."""
    parsed = pd.to_datetime(date_series, errors='coerce', format='mixed')
    mask = parsed.isna()
    if mask.any():
        from datetime import datetime
        current_year = datetime.now().year
        # Handle '01 Feb' by padding it up with the current year
        parsed[mask] = pd.to_datetime(date_series[mask].astype(str) + f" {current_year}", format="mixed", errors="coerce")
    return parsed

# --- Tab 1: Statement Analysis ---
with tab1:
    if uploaded_file is not None and analyze_button:
        with st.spinner("Parsing document..."):
            if uploaded_file.name.endswith(".pdf"):
                raw_txns = parse_pdf(uploaded_file)
            else:
                raw_txns = parse_csv(uploaded_file)
                
            if not raw_txns:
                st.error("No transactions found or error parsing document.")
            else:
                st.session_state.transactions = categorize_all(raw_txns, response_mode)
                
                # Fetch text analysis from LLM on new upload and save immediately
                with st.spinner("Generating AI Analysis of your spending..."):
                    st.session_state.analysis_text = generate_financial_analysis(st.session_state.transactions)
                
    if st.session_state.transactions:
        txns = st.session_state.transactions
        df = pd.DataFrame(txns)
        
        display_df = pd.DataFrame({
            "Date": df.get("date", []),
            "Description": df.get("description", []),
            "Amount": df.get("amount", []),
            "Category": df.get("category", []),
            "Confidence": [format_confidence(l) for l in df.get("confidence_label", [])],
        })
        
        if "model_votes" in df.columns:
            def format_votes(votes):
                if not votes or len(votes) != 1:
                    return str(votes)
                return f"Groq: {votes[0]}"
            display_df["Model Votes"] = df["model_votes"].apply(format_votes)

        total_spend = display_df["Amount"].sum()
        top_category = display_df["Category"].mode()[0] if not display_df.empty else "N/A"
        low_conf_count = len([c for c in display_df["Confidence"] if "LOW" in c])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Spend", f"${total_spend:.2f}")
        col2.metric("Top Category", top_category)
        col3.metric("Flagged Info (Low Conf)", low_conf_count)
        
        # Display AI Narrative Insights
        if "analysis_text" in st.session_state and st.session_state.analysis_text:
            with st.container(border=True):
                st.subheader("🤖 AI Diagnostic Output")
                st.markdown(st.session_state.analysis_text)
        
        if "parsed_date" not in df.columns:
            df["parsed_date"] = robust_parse_dates(df["date"])

        # Create dictionaries to store created figures universally for PDF passing later
        if "saved_figs" not in st.session_state:
            st.session_state.saved_figs = {}

        st.subheader("Spend by Category")
        cat_spend = df.groupby("category")["amount"].sum().reset_index()
        fig_bar = px.bar(cat_spend, x="category", y="amount", text="amount", color="category", template="plotly_dark", title="Total Spend per Category")
        fig_bar.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        st.plotly_chart(fig_bar, width="stretch")
        st.session_state.saved_figs["Spend By Category (Bar)"] = fig_bar
        
        col_pie, col_sun = st.columns(2)
        with col_pie:
            st.subheader("Category Distribution")
            fig_pie = px.pie(cat_spend, values="amount", names="category", template="plotly_dark", hole=0.4, title="Category Share")
            st.plotly_chart(fig_pie, width="stretch")
            st.session_state.saved_figs["Category Split (Donut)"] = fig_pie
            
        with col_sun:
            st.subheader("Hierarchical Breakdown")
            fig_sun = px.sunburst(df, path=['category', 'description'], values='amount', template="plotly_dark", title="Detailed Breakdown (Interactive)")
            # Fix text layout to avoid overlapping
            fig_sun.update_traces(textinfo="label+percent parent")
            st.plotly_chart(fig_sun, width="stretch")
            st.session_state.saved_figs["Hierarchical Spend Breakdown (Sunburst)"] = fig_sun


# --- Tab 2: Chat with your data ---
with tab2:
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])
        
    if prompt := st.chat_input("Ask about your finances..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with chat_container:
            st.chat_message("user").write(prompt)
        
        rag_context = retrieve_context(prompt, index, chunks) if index else ""
        
        live_search_keywords = [
            "current", "today", "benchmark", "inflation", "interest rate", 
            "budget", "news", "2024", "2025", "2026", "latest", "new", 
            "tax", "market", "policy", "update"
        ]
        search_result = ""
        if any(kw in prompt.lower() for kw in live_search_keywords):
            search_result = tavily_search(prompt)
            
        txn_summary = "No transactions analyzed yet."
        if st.session_state.transactions:
            txn_summary = pd.DataFrame(st.session_state.transactions).to_string(index=False)
            
        system_instruction = "Give a thorough explanation with specific numbers from the transactions, practical tips, and a breakdown."
        if response_mode == "Concise":
            system_instruction = "Reply in 2-3 sentences maximum. Be direct."
            
        system_prompt = f"""You are FinBot, an expert financial advisor. 
{system_instruction}

User Transactions context:
{txn_summary[:3000]} # Trim to fit safely

Knowledge Base context:
{rag_context}

Live Search context:
{search_result}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        with chat_container:
            with st.spinner("Generating answer..."):
                answer = get_chat_response(messages, llm_provider)
            st.chat_message("assistant").write(answer)
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


# --- Tab 3: Insights ---
with tab3:
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions)
        
        if "parsed_date" not in df.columns:
            df["parsed_date"] = robust_parse_dates(df["date"])
            
        trend_df = df.dropna(subset=["parsed_date"]).sort_values("parsed_date")
        
        if not trend_df.empty:
            daily_spend = trend_df.groupby("parsed_date")["amount"].sum().reset_index()
            fig_line = px.line(daily_spend, x="parsed_date", y="amount", markers=True, template="plotly_dark", title="Daily Spending Velocity")
            st.plotly_chart(fig_line, width="stretch")
            st.session_state.saved_figs["Spending Velocity Trends (Line)"] = fig_line
        else:
            st.info("Could not parse enough dates to show a trend line.")
        
        st.subheader("Transaction Anomaly Analysis")
        # A boxplot visualization is brilliant to find outlier transactions statistically.
        fig_box = px.box(df, x="category", y="amount", color="category", template="plotly_dark", title="Spend Distribution Points & Top Anomalies")
        st.plotly_chart(fig_box, width="stretch")
        st.session_state.saved_figs["Spend Distribution (Boxplot)"] = fig_box
        
        st.subheader("Top 5 Largest Transactions")
        top5 = df.nlargest(5, "amount")[["date", "description", "amount", "category"]]
        st.table(top5)
        
        st.subheader("Recurring Transactions")
        recurring = df.groupby('description').filter(lambda x: len(x) > 1)
        if not recurring.empty:
            st.dataframe(recurring[["date", "description", "amount", "category"]])
        else:
            st.info("No recurring transactions detected.")
            
        st.subheader("Anomaly Alerts")
        cat_averages = df.groupby("category")["amount"].mean()
        anomalies = []
        for _, row in df.iterrows():
            avg = cat_averages.get(row["category"], 0)
            if avg > 0 and row["amount"] > 2 * avg:
                anomalies.append(row)
                
                
        if anomalies:
            st.warning(f"Found {len(anomalies)} transactions double the typical category average.")
            st.dataframe(pd.DataFrame(anomalies)[["date", "description", "amount", "category"]])
        else:
            st.success("No significant anomalies detected.")

# --- Sidebar PDF Export (Always reachable if data exists) ---
if st.session_state.transactions and "saved_figs" in st.session_state and st.session_state.saved_figs:
    st.sidebar.divider()
    st.sidebar.header("Export")
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        pdf_path = tmp_pdf.name
        
    if st.sidebar.button("Generate Downloadable PDF Report"):
        with st.spinner("Compiling professional PDF report with charts and insights..."):
            a_text = st.session_state.analysis_text if "analysis_text" in st.session_state else ""
            generate_pdf_report(pd.DataFrame(st.session_state.transactions), st.session_state.saved_figs, a_text, pdf_path)
        
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
            
        st.sidebar.download_button(
            label="⬇️ Download Your FinBot PDF",
            data=pdf_bytes,
            file_name="FinBot_Intelligence_Report.pdf",
            mime="application/pdf",
        )
