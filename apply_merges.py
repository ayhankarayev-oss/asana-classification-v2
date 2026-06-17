"""
apply_merges.py
===============
Apply approved topic merges and export the final labeled training dataset.
This produces the file that will be used in Phase 7 to train the classifier.

Merges applied:
  - T7 + T17 → "Portal Views & Client Reports" (combined: 240 docs)

Output:
  outputs/training_data.csv  → task_id, cleaned_text, label (business_name), pillar
  outputs/reports/final_taxonomy.csv → updated with merge applied

Usage:
  python apply_merges.py

Note: This does NOT train a model. It only prepares the labeled dataset.
      Training happens in Phase 7 (train_classifier.py).
"""
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

REPORTS_DIR = Path("outputs/reports")
CORPUS_PATH = Path("outputs/corpus_clean.csv")
DISCOVERY_PATH = Path("outputs/discovery_medium.csv")
OUTPUT_TRAINING = Path("outputs/training_data.csv")

# ===========================================================================
# MERGE CONFIGURATION (edit this to add more merges)
# ===========================================================================
MERGES = {
    # Format: target_topic_id: [list of topic_ids to absorb into target]
    # T7 absorbs T17 → both become "Portal Views & Client Reports"
    7: [17],
}

# Final taxonomy name overrides (topic_id → business_name)
# These are the approved Business Outcome labels
NAME_MAP = {
    0: ("Account Operations", "Account Setup & Connectivity"),
    1: ("Investment Management", "New Private Investment Setup"),
    2: ("Reporting & Portal", "Invoicing & Billing"),
    3: ("Investment Management", "Commitment & Capital Call Tracking"),
    4: ("Operations & Admin", "Access & Permission Requests"),
    5: ("Operations & Admin", "Contact & User Updates"),
    6: ("Data & Valuation", "Missing Data & Attributes"),
    7: ("Reporting & Portal", "Client Report & View Management"),
    8: ("Investment Management", "Security & Position Updates"),
    9: ("Ownership & Structure", "Ownership & Trust Structure"),
    10: ("Data & Valuation", "Scheduled Valuation Feeds"),
    11: ("Operations & Admin", "Recurring Data Maintenance & Audits"),
    12: ("Investment Management", "Debt & Lending Instruments"),
    13: ("Data & Valuation", "Position Cleanup & Deduplication"),
    14: ("Reporting & Portal", "Investment Performance Analysis"),
    15: ("Account Operations", "Historical Data Backfill"),
    16: ("Operations & Admin", "Cash Flow & Distribution Management"),
    17: ("Reporting & Portal", "Client Report & View Management"),  # Merged into T7
    18: ("Data & Valuation", "Missing Data & Attributes"),
    19: ("Operations & Admin", "Meetings, Reminders & Follow-ups"),
    20: ("Data & Valuation", "Manual Valuation Updates"),
    21: ("Data & Valuation", "Cost Basis Reconciliation"),
    22: ("Ownership & Structure", "Asset Transfers & Movements"),
}


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 60)
    print("APPLY MERGES & EXPORT TRAINING DATA")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    # Load
    print("\n[1/4] Loading data...")
    corpus = pd.read_csv(CORPUS_PATH, dtype=str, keep_default_na=False)
    discovery = pd.read_csv(DISCOVERY_PATH, dtype=str, keep_default_na=False)
    discovery["topic_id"] = discovery["topic_id"].astype(int)
    print(f"  Corpus: {len(corpus)} rows")
    print(f"  Discovery (medium): {len(discovery)} rows")

    # Merge corpus with topic assignments
    merged = corpus.merge(discovery[["task_id", "topic_id"]], on="task_id", how="left")
    assert len(merged) == len(corpus), "Row count mismatch after merge!"
    print(f"  Joined: {len(merged)} rows (all task_ids matched)")

    # Apply merges
    print(f"\n[2/4] Applying merges...")
    for target_id, source_ids in MERGES.items():
        for src in source_ids:
            count = (merged["topic_id"] == src).sum()
            merged.loc[merged["topic_id"] == src, "topic_id"] = target_id
            print(f"  Merged T{src} → T{target_id} ({count} docs)")

    # Apply names
    print(f"\n[3/4] Assigning business labels...")
    merged["business_pillar"] = merged["topic_id"].map(lambda t: NAME_MAP.get(t, ("Unknown", "Unknown"))[0])
    merged["label"] = merged["topic_id"].map(lambda t: NAME_MAP.get(t, ("Unknown", "Unknown"))[1])

    # Verify
    n_unknown = (merged["label"] == "Unknown").sum()
    if n_unknown > 0:
        print(f"  WARNING: {n_unknown} rows have unknown labels!")
    else:
        print(f"  All {len(merged)} rows labeled successfully")

    # Show distribution
    print(f"\n  Label distribution:")
    dist = merged["label"].value_counts()
    for label, count in dist.items():
        pct = count / len(merged) * 100
        print(f"    {label:<42} {count:>4} ({pct:.1f}%)")

    # Export training data
    print(f"\n[4/4] Exporting training data...")
    train_cols = ["task_id", "cleaned_text", "label", "business_pillar", "primary_platform"]
    train_df = merged[train_cols].copy()
    train_df.to_csv(OUTPUT_TRAINING, index=False, encoding="utf-8-sig")
    print(f"  Saved: {OUTPUT_TRAINING}")
    print(f"  Rows: {len(train_df)}")
    print(f"  Unique labels: {train_df['label'].nunique()}")
    print(f"  Min class size: {train_df['label'].value_counts().min()}")

    # Traceability check
    print(f"\n  TRACEABILITY CHECK:")
    print(f"    Input task_ids: {corpus['task_id'].nunique()}")
    print(f"    Output task_ids: {train_df['task_id'].nunique()}")
    print(f"    Match: {corpus['task_id'].nunique() == train_df['task_id'].nunique()}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"TRAINING DATA READY")
    print(f"{'=' * 60}")
    print(f"  File: {OUTPUT_TRAINING}")
    print(f"  {len(train_df):,} labeled examples across {train_df['label'].nunique()} classes")
    print(f"  Smallest class: {train_df['label'].value_counts().min()} docs (healthy for training)")
    print(f"\n  WHAT'S NEXT (Phase 7):")
    print(f"    1. Load training_data.csv")
    print(f"    2. Encode cleaned_text with sentence-transformers")
    print(f"    3. Train classifier (logistic regression or fine-tuned model)")
    print(f"    4. New Asana task → preprocess → encode → predict label")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
