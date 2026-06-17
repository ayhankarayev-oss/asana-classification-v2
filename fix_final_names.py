"""
fix_final_names.py
===================
Updates training_data.csv with the correct FINAL cluster names and pillars
that were decided during the taxonomy refinement process.

Also updates the Sub-type Definitions section in the dashboard.
"""
import pandas as pd
from pathlib import Path

TRAINING_PATH = Path("outputs/training_data.csv")

# Final name corrections: old_name -> new_name
NAME_FIXES = {
    "New Account Onboarding": "Account Setup & Connectivity",
    "Distribution & Income Recording": "Cash Flow & Distribution Management",
    "Unclassified (Topic 20)": "Periodic NAV & Valuation Entry",
}

# Final pillar corrections (these change too)
PILLAR_FIXES = {
    "Account Setup & Connectivity": "Account & Data Onboarding",
    "Cash Flow & Distribution Management": "Transaction & Cash Flow Processing",
    "Periodic NAV & Valuation Entry": "Valuation & Pricing",
}


def main():
    print("=" * 60)
    print("FIX FINAL NAMES IN training_data.csv")
    print("=" * 60)

    train = pd.read_csv(TRAINING_PATH, dtype=str, keep_default_na=False)
    print(f"\nBefore fix: {train['label'].nunique()} unique labels")

    # Apply name fixes
    for old_name, new_name in NAME_FIXES.items():
        count = (train["label"] == old_name).sum()
        if count > 0:
            train.loc[train["label"] == old_name, "label"] = new_name
            print(f"  Renamed: '{old_name}' -> '{new_name}' ({count} docs)")
        else:
            print(f"  SKIP (not found): '{old_name}'")

    # Fix pillars for renamed classes
    for label, pillar in PILLAR_FIXES.items():
        mask = train["label"] == label
        if mask.sum() > 0:
            train.loc[mask, "business_pillar"] = pillar

    # Verify
    print(f"\nAfter fix: {train['label'].nunique()} unique labels")
    print("\nFinal labels:")
    for label in sorted(train["label"].unique()):
        count = (train["label"] == label).sum()
        pillar = train[train["label"] == label]["business_pillar"].iloc[0]
        print(f"  {label:<42} {count:>4}  ({pillar})")

    # Save
    train.to_csv(TRAINING_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {TRAINING_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
