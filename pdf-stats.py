"""
pdf-stats.py – checks whether each PDF in a directory
contains extractable text or is likely a scan (image-only).
Writes results to pdf_stats.csv.
"""

import os
import pdfplumber
import pandas as pd

# ─────────── adjust this to where your PDFs live ───────────
PDF_DIR = "pdf_okw"
# ─────────────────────────────────────────────────────────────

results = []
for fname in sorted(os.listdir(PDF_DIR)):
    if not fname.lower().endswith(".pdf"):
        continue
    path = os.path.join(PDF_DIR, fname)
    chars = 0
    try:
        with pdfplumber.open(path) as pdf:
            # sample up to first 3 pages
            for page in pdf.pages[:3]:
                text = page.extract_text() or ""
                chars += len(text)
        pdf_type = "text" if chars > 100 else "scan"
    except Exception as e:
        pdf_type = "error"
        chars = 0

    results.append({
        "file": fname,
        "chars_sample": chars,
        "type": pdf_type,
    })

df = pd.DataFrame(results)
# Save to CSV for inspection:
out_csv = os.path.join(PDF_DIR, "pdf_stats.csv")
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"Wrote stats for {len(df)} files to {out_csv}\n")
print(df.head(20).to_string(index=False))




#results = []
#for fname in os.listdir(PDF_DIR):
#    if not fname.lower().endswith(".pdf"):
#        continue
#    path = os.path.join(PDF_DIR, fname)
#    try:
#        with pdfplumber.open(path) as pdf:
#            # Count total characters across first 3 pages (or all pages if fewer)
#            chars = 0
#            for page in pdf.pages[:3]:
#                text = page.extract_text() or ""
#                chars += len(text)
#        pdf_type = "text" if chars > 100 else "scan"
#    except Exception as e:
#        pdf_type = f"error: {e}"
#        chars = 0
#    results.append({"file": fname, "chars_sample": chars, "type": pdf_type})
#
## Present results in a DataFrame
#df = pd.DataFrame(results).sort_values("type")
#import ace_tools as tools; tools.display_dataframe_to_user("PDF Text vs Scan", df)
