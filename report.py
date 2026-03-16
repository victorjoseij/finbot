import os
import tempfile
from fpdf import FPDF
import pandas as pd

class PDFReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 16)
        self.cell(180, 10, 'FinBot - Personal Finance Intelligence Report', 0, 1, 'C')
        self.set_line_width(0.5)
        self.line(10, 22, 200, 22)
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(180, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(df: pd.DataFrame, figs: dict, analysis_text: str, output_path: str):
    """Generates a professional PDF containing key financial insights and all provided plotly figures."""
    pdf = PDFReport()
    pdf.set_margins(left=15, top=15, right=15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # 1. Executive Summary
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(180, 10, 'Executive Summary', 0, 1)
    
    pdf.set_font('helvetica', '', 12)
    total_spend = df["amount"].sum()
    top_cat = df["category"].mode()[0] if not df.empty else "N/A"
    total_txns = len(df)
    
    pdf.multi_cell(180, 7, f"- Total Spend: INR {total_spend:,.2f}  (or your local currency)")
    pdf.multi_cell(180, 7, f"- Top Spending Category: {top_cat}")
    pdf.multi_cell(180, 7, f"- Total Transactions Analyzed: {total_txns}")
    pdf.ln(10)
    
    if analysis_text:
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(180, 10, 'AI Diagnostic Analysis', 0, 1)
        pdf.set_font('helvetica', '', 11)
        # Avoid FPDF latin-1 breaking by enforcing ascii/latin-1 safety dynamically
        safe_text = analysis_text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(180, 6, safe_text)
        pdf.ln(10)

    # 2. Key Anomalies
    cat_averages = df.groupby("category")["amount"].mean()
    anomalies = []
    for _, row in df.iterrows():
        avg = cat_averages.get(row["category"], 0)
        if avg > 0 and row["amount"] > 2 * avg:
            anomalies.append(row)
            
    if anomalies:
        pdf.set_font('helvetica', 'B', 12)
        pdf.set_text_color(220, 50, 50) # Red color for alerts
        pdf.multi_cell(180, 7, f"Alert: Detected {len(anomalies)} transactions that are double their category average.")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        
    # 3. Visualizations
    for fig_name, fig in figs.items():
        pdf.add_page()
        pdf.set_font("helvetica", 'B', 14)
        pdf.cell(180, 10, fig_name, 0, 1, 'C')
        pdf.ln(5)
        
        # Save temp image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_name = tmp.name
            
        try:
            # We enforce standard layout size for high-res PDF import
            fig.write_image(tmp_name, width=900, height=600, scale=2)
            pdf.image(tmp_name, x=15, w=180)
        except Exception as e:
            pdf.set_font("helvetica", 'I', 10)
            pdf.cell(180, 10, f"[Could not render graph natively: {e}]", 0, 1)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
                
    pdf.output(output_path)
