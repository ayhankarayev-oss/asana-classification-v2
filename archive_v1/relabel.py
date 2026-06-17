"""
Quick-patch script: re-applies updated BUSINESS_LABELS to complaints_classified.csv
and regenerates cluster_summary_report.md header table + executive summary.
Run this whenever you change BUSINESS_LABELS without wanting to re-run full BERTopic.
"""
import pandas as pd
from pathlib import Path
from collections import Counter

BUSINESS_LABELS = {
    -1: "Outliers / Unclassified",
     0: "New Account & Data Feed Setup",
     1: "Portfolio Platform Account Updates",
     2: "New Private Investment Entry",
     3: "Private Investment Updates & Valuations",
     4: "Trust & Estate Cash Flow Verification",
     5: "Capital Call Audit & Monthly Statement Review",
     6: "Cost Basis & Data Quality Fixes",
     7: "Document Upload & Client Billing",
     8: "Reporting & Performance Analytics",
     9: "Ownership Structure & Legal Entity Changes",
    10: "Real Asset Transaction Audit",
    11: "Loan & Lending Account Setup",
    12: "Direct Deal Updates & Unlinked Accounts",
    13: "Portfolio Platform View & Access Configuration",
    14: "General & Ad Hoc Requests",
}

CSV_PATH    = Path("complaints_classified.csv")
EXPL_PATH   = Path("topic_explanations.csv")
REPORT_PATH = Path("cluster_summary_report.md")

# --- Patch complaints_classified.csv ---
print(f"Loading {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
df["topic_id"] = pd.to_numeric(df["topic_id"], errors="coerce").fillna(-1).astype(int)
df["business_label"] = df["topic_id"].map(BUSINESS_LABELS).fillna("Outliers / Unclassified")
df.to_csv(CSV_PATH, index=False)
print(f"  Patched {len(df):,} rows -> {CSV_PATH}")

# --- Patch topic_explanations.csv ---
if EXPL_PATH.exists():
    expl = pd.read_csv(EXPL_PATH, dtype=str, keep_default_na=False)
    expl["topic_id"] = pd.to_numeric(expl["topic_id"], errors="coerce").fillna(-1).astype(int)
    expl["business_label"] = expl["topic_id"].map(BUSINESS_LABELS).fillna("Outliers / Unclassified")
    expl.to_csv(EXPL_PATH, index=False)
    print(f"  Patched {len(expl):,} rows -> {EXPL_PATH}")

# --- Print updated distribution for verification ---
print("\nUpdated label distribution:")
print(f"  {'Topic':>6}  {'Count':>6}  {'%':>6}  Label")
print("  " + "-" * 70)
total = len(df)
for tid in sorted(BUSINESS_LABELS.keys()):
    cnt = int((df["topic_id"] == tid).sum())
    if cnt > 0:
        label = BUSINESS_LABELS[tid]
        pct = cnt / total * 100
        print(f"  {tid:>6}  {cnt:>6,}  {pct:>5.1f}%  {label}")

# --- Update cluster_summary_report.md header table ---
if REPORT_PATH.exists():
    text = REPORT_PATH.read_text(encoding="utf-8")
    # Update topic headings in the report (replace old label -- same topic ID)
    for tid, new_label in BUSINESS_LABELS.items():
        if tid == -1:
            continue
        # Match lines like: ## Topic N -- Old Label Name
        import re
        pattern = rf"(## Topic {tid} -- )([^\n]+)"
        replacement = rf"\1{new_label}"
        text = re.sub(pattern, replacement, text)
    # Also fix the Cluster ID lines in Operational Summary
    # (they just show the ID, no label to fix there)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"\nUpdated topic headings in {REPORT_PATH}")

print("\nRelabeling complete.")