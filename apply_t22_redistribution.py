"""
T22 Redistribution: Apply manual corrections + regenerate reports.
Uses KNN base assignments with semantic overrides for the ~36% that KNN got wrong.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from pathlib import Path

TRAINING_PATH = Path(r"c:\Users\Lenovo\Desktop\asana-classification-v2\outputs\training_data.csv")
EMBEDDINGS_PATH = Path(r"c:\Users\Lenovo\Desktop\asana-classification-v2\outputs\embeddings.npy")
CORPUS_PATH = Path(r"c:\Users\Lenovo\Desktop\asana-classification-v2\outputs\corpus_clean.csv")
REPORT_PATH = Path(r"c:\Users\Lenovo\Desktop\asana-classification-v2\outputs\reports\leadership_report.html")

# Manual semantic overrides (GID -> correct class based on review)
OVERRIDES = {
    # From "Missing Data" - wrong assignments
    "1213972391893430": "New Private Investment Setup",          # pari-passu investment = new investment
    "1213730767354433": "Security & Position Updates",          # coreshell series b = security update
    "1212840690205420": "Security & Position Updates",          # nona master sold for 4m = exit event
    "1207859466652859": "Periodic NAV & Valuation Entry",       # fund quarterly update = valuation
    # From "Account Setup" - wrong assignments
    "1214068138663264": "Recurring Data Maintenance & Audits",  # investor account statement = recurring
    "1214979150436511": "Missing Data & Attributes",            # process 5 files = data processing
    "1213334714616879": "Periodic NAV & Valuation Entry",       # update values of central park = valuation
    "1211884664025259": "Debt & Lending Instruments",           # update mortgages = debt
    "1211276528150376": "Security & Position Updates",          # career karma to outrival = security rename
    "1211066386370463": "Client Report & View Management",      # updates to balance sheet view = reporting
    # From "Recurring Maintenance" - wrong assignments
    "1215181996468061": "Missing Data & Attributes",            # frost brokerage statement = data processing
    "1212323360901948": "Client Report & View Management",      # tax loss harvesting report = report creation
    # From "Position Cleanup" - wrong
    "1207206951248877": "Access & Permission Requests",         # remove aditya nagar access = access mgmt
    # From "New Private Investment" - wrong
    "1213266878099214": "Missing Data & Attributes",            # cherry family drop update = data update
    "1211862669241638": "Position Cleanup & Deduplication",     # remove albanese real estate = position removal
    # From "Commitment" - wrong
    "1214052159835054": "Periodic NAV & Valuation Entry",       # process valuation updates from files = valuation
    "1214079549751631": "Ownership & Trust Structure",          # add entity id and direct owner ids = ownership
    # From "Distribution" - wrong
    "1212950893095891": "Ownership & Trust Structure",          # moved from lllp to irrevocable = trust transfer
}

print("=" * 70)
print("T22 REDISTRIBUTION: KNN + Semantic Overrides")
print("=" * 70)

# Load
corpus = pd.read_csv(CORPUS_PATH, dtype=str, keep_default_na=False)
train = pd.read_csv(TRAINING_PATH, dtype=str, keep_default_na=False)
embeddings = np.load(str(EMBEDDINGS_PATH))

label_map = dict(zip(train["task_id"], train["label"]))
labels = [label_map.get(tid, "Unknown") for tid in corpus["task_id"]]
texts = corpus["cleaned_text"].tolist()
task_ids = corpus["task_id"].tolist()

t22_label = "Asset Transfers & Movements"
t22_indices = [i for i, l in enumerate(labels) if l == t22_label]
other_indices = [i for i, l in enumerate(labels) if l != t22_label]

print(f"\nT22 docs to redistribute: {len(t22_indices)}")

# KNN base assignment (K=7)
t22_embeddings = embeddings[t22_indices]
other_embeddings = embeddings[other_indices]
other_labels = [labels[i] for i in other_indices]

sim_matrix = cosine_similarity(t22_embeddings, other_embeddings)

assignments = {}
for idx, t22_idx in enumerate(t22_indices):
    tid = task_ids[t22_idx]
    sims = sim_matrix[idx]
    top_k = np.argsort(sims)[-7:][::-1]
    top_labels = [other_labels[p] for p in top_k]
    vote = Counter(top_labels).most_common(1)[0][0]
    
    # Apply override if exists
    if tid in OVERRIDES:
        assignments[tid] = OVERRIDES[tid]
    else:
        assignments[tid] = vote

# Update training data
print("\n[APPLYING] Updating training_data.csv...")
for tid, new_label in assignments.items():
    train.loc[train["task_id"] == tid, "label"] = new_label

# Remove T22 as a class entirely
remaining_t22 = train[train["label"] == t22_label]
if len(remaining_t22) > 0:
    print(f"  WARNING: {len(remaining_t22)} docs still labeled as T22!")
else:
    print(f"  T22 class eliminated (0 docs remaining)")

# Save updated training data
train.to_csv(TRAINING_PATH, index=False, encoding="utf-8-sig")
print(f"  Saved: {TRAINING_PATH}")

# Distribution summary
print(f"\n[RESULT] New label distribution after T22 redistribution:")
dist = train["label"].value_counts()
print(f"  Classes: {len(dist)}")
print(f"  Total docs: {len(train)}")
for label, count in dist.items():
    print(f"    {label:<42} {count:>4} ({count/len(train)*100:.1f}%)")

# Redistribution report
print(f"\n[REDISTRIBUTION LOG]")
dest_counts = Counter(assignments.values())
for dest, count in dest_counts.most_common():
    overridden = sum(1 for tid, lbl in assignments.items() if lbl == dest and tid in OVERRIDES)
    print(f"  → {dest:<42} {count:>3} docs ({overridden} manually corrected)")
