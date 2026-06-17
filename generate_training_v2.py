"""
generate_training_v2.py
========================
Applies v2 transforms to training_data.csv and exports training_data_v2.csv.
V2 changes:
- Merge sub-types: Security & Position Updates + Position Cleanup & Deduplication -> Security & Position Management
- Regroup pillars: Cash Flow -> Capital & Cash Flow Management, Commitment -> same, Debt -> Portfolio, Ownership standalone
"""
import pandas as pd
from pathlib import Path

TRAIN_PATH = Path("outputs/training_data.csv")
OUTPUT_PATH = Path("outputs/training_data_v2.csv")

# Sub-type merge
SUBTYPE_MERGE = {
    "Security & Position Updates": "Security & Position Management",
    "Position Cleanup & Deduplication": "Security & Position Management",
}

# Issue Type reassignments
PILLAR_OVERRIDES = {
    "Cash Flow & Distribution Management": "Capital & Cash Flow Management",
    "Commitment & Capital Call Tracking": "Capital & Cash Flow Management",
    "Debt & Lending Instruments": "Portfolio & Investment Operations",
    "Ownership & Trust Structure": "Ownership & Trust Structure",
    "Security & Position Management": "Portfolio & Investment Operations",
}


def main():
    df = pd.read_csv(TRAIN_PATH, dtype=str, keep_default_na=False)
    print(f"Loaded: {len(df)} rows, {df['label'].nunique()} labels, {df['business_pillar'].nunique()} pillars")

    # Step 1: Merge sub-types
    df["label"] = df["label"].replace(SUBTYPE_MERGE)

    # Step 2: Reassign pillars for affected labels
    for label, pillar in PILLAR_OVERRIDES.items():
        mask = df["label"] == label
        if mask.any():
            df.loc[mask, "business_pillar"] = pillar

    print(f"After v2: {len(df)} rows, {df['label'].nunique()} labels, {df['business_pillar'].nunique()} pillars")
    print("\nLabel distribution:")
    for lbl in sorted(df["label"].unique()):
        count = (df["label"] == lbl).sum()
        pillar = df[df["label"] == lbl]["business_pillar"].iloc[0]
        print(f"  {lbl:<40} {count:>4}  ({pillar})")

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
