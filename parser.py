import re
import logging
import pdfplumber
import pandas as pd
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def parse_pdf(file) -> List[Dict[str, Any]]:
    """Parses a bank statement PDF and extracts transactions."""
    transactions = []
    # Regex to handle formats like DD MMM (e.g., "01 Feb") or YYYY-MM-DD
    # Followed by description (with optional ref no before it), and eventually amounts like 1,842.00 or 28,000.00
    date_pattern = re.compile(r"(\d{1,2}\s+[A-Za-z]{3}|\d{2,4}[-/]\d{2}[-/]\d{2,4})\s+(.+?)\s+([\d,]+\.\d{2})")
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    for line in text.split('\n'):
                        match = date_pattern.search(line)
                        if match:
                            date_str = match.group(1).strip()
                            desc = match.group(2).strip()
                            amount_str = match.group(3).strip().replace(',', '')
                            
                            try:
                                amount = float(amount_str)
                                transactions.append({
                                    "date": date_str,
                                    "description": desc,
                                    "amount": amount,
                                    "type": "debit" # Approximation for basic parsed CSVs
                                })
                            except ValueError:
                                continue
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
    return transactions

def parse_csv(file) -> List[Dict[str, Any]]:
    """Parses a bank statement CSV and extracts transactions."""
    transactions = []
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.astype(str).str.lower().str.strip()
        
        # Heuristics to identify relevant columns
        date_col = next((col for col in df.columns if 'date' in col), None)
        desc_col = next((col for col in df.columns if 'desc' in col or 'detail' in col or 'memo' in col), None)
        amount_col = next((col for col in df.columns if 'amount' in col or 'value' in col), None)
        
        if not (date_col and desc_col and amount_col):
            logger.warning("Could not automatically detect all normal matching columns in CSV (date, description, amount).")
            if len(df.columns) >= 3:
                date_col, desc_col, amount_col = df.columns[0:3]
            else:
                return []

        for _, row in df.iterrows():
            try:
                date_str = str(row[date_col]).strip()
                desc_str = str(row[desc_col]).strip()
                
                amount_raw = str(row[amount_col]).replace(',', '').strip()
                if amount_raw.startswith('$'):
                    amount_raw = amount_raw[1:]
                amount = float(amount_raw)
                
                transactions.append({
                    "date": date_str,
                    "description": desc_str,
                    "amount": amount,
                    "type": "debit" if amount < 0 else "credit"
                })
            except Exception as row_error:
                continue
                
    except Exception as e:
        logger.error(f"Error parsing CSV: {e}")
        
    return transactions
