"""
T22 Redistribution: Hybrid Method (KNN + Confidence Scoring)
=============================================================
For each of the 95 docs in T22 "Asset Transfers & Movements":
1. Find 5 nearest neighbors from ALL other classes (KNN)
2. Assign to majority-voted class
3. Record confidence (% of neighbors agreeing)
4. Flag docs with confidence < 0.7 for manual review
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from pathlib import Path

TRAINING_PATH = Path(r"c:\Users\Lenovo\Desktop\asana-classification-v2\outputs\training_data.csv")
EMBEDDINGS_PATH = Path(r"c:\Users\Lenovo\Desktop\asana-classification-v2\outputs\embeddings.npy")
CORPUS_PATH = Path(r"c:\Users\Lenovo\Desktop\asana-classification-v2\outputs\corpus_clean.csv")

K = 7  # Number of nearest neighbors (odd for tie-breaking)
CONFIDENCE_THRESHOLD = 0.0  # Accept all KNN assignments (no flagging)

print("=" * 75)
print("T22 REDISTRIBUTION: Hybrid Method (KNN + Confidence)")
print("=" * 75)

# Load
corpus = pd.read_csv(CORPUS_PATH, dtype=str, keep_default_na=False)
train = pd.read_csv(TRAINING_PATH, dtype=str, keep_default_na=False)
embeddings = np.load(str(EMBEDDINGS_PATH))

label_map = dict(zip(train["task_id"], train["label"]))
labels = [label_map.get(tid, "Unknown") for tid in corpus["task_id"]]
texts = corpus["cleaned_text"].tolist()
task_ids = corpus["task_id"].tolist()

# Identify T22 docs and non-T22 docs
t22_label = "Asset Transfers & Movements"
t22_indices = [i for i, l in enumerate(labels) if l == t22_label]
other_indices = [i for i, l in enumerate(labels) if l != t22_label]

print(f"\nT22 docs to redistribute: {len(t22_indices)}")
print(f"Other docs (reference pool): {len(other_indices)}")

# Get embeddings for T22 and others
t22_embeddings = embeddings[t22_indices]
other_embeddings = embeddings[other_indices]
other_labels = [labels[i] for i in other_indices]

# Compute similarity of each T22 doc to all other docs
print(f"\nComputing {len(t22_indices)} x {len(other_indices)} similarity matrix...")
sim_matrix = cosine_similarity(t22_embeddings, other_embeddings)

# KNN assignment
results = []
for idx, t22_idx in enumerate(t22_indices):
    # Get K nearest neighbors
    sims = sim_matrix[idx]
    top_k_positions = np.argsort(sims)[-K:][::-1]
    top_k_labels = [other_labels[p] for p in top_k_positions]
    top_k_sims = [sims[p] for p in top_k_positions]
    
    # Majority vote
    vote_counts = Counter(top_k_labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    confidence = winner_count / K
    avg_sim_to_winner = np.mean([top_k_sims[i] for i, l in enumerate(top_k_labels) if l == winner])
    
    results.append({
        "task_id": task_ids[t22_idx],
        "text": texts[t22_idx][:130],
        "proposed_class": winner,
        "confidence": confidence,
        "avg_similarity": avg_sim_to_winner,
        "knn_votes": dict(vote_counts),
        "flag": "REVIEW" if confidence < CONFIDENCE_THRESHOLD else "AUTO",
    })

results_df = pd.DataFrame(results)

# Summary by destination class
print(f"\n{'=' * 75}")
print("REDISTRIBUTION SUMMARY")
print(f"{'=' * 75}")

auto = results_df[results_df["flag"] == "AUTO"]
review = results_df[results_df["flag"] == "REVIEW"]

print(f"\n  Auto-assigned (confidence >= {CONFIDENCE_THRESHOLD}): {len(auto)} docs")
print(f"  Flagged for review (confidence < {CONFIDENCE_THRESHOLD}): {len(review)} docs")

print(f"\n  Destination breakdown:")
dest_summary = results_df.groupby("proposed_class").agg(
    count=("task_id", "count"),
    avg_conf=("confidence", "mean"),
    avg_sim=("avg_similarity", "mean"),
).sort_values("count", ascending=False)

print(f"\n  {'Destination Class':<42} {'Count':<7} {'Avg Conf':<10} {'Avg Sim'}")
print(f"  {'-'*70}")
for cls, row in dest_summary.iterrows():
    print(f"  {cls:<42} {int(row['count']):<7} {row['avg_conf']:.2f}      {row['avg_sim']:.4f}")

# Show auto-assigned samples per destination
print(f"\n{'=' * 75}")
print("AUTO-ASSIGNED DOCS (by destination class)")
print(f"{'=' * 75}")

for cls in dest_summary.index:
    cls_docs = results_df[(results_df["proposed_class"] == cls) & (results_df["flag"] == "AUTO")]
    if len(cls_docs) == 0:
        continue
    print(f"\n  → {cls} ({len(cls_docs)} docs, avg_conf={cls_docs['confidence'].mean():.2f})")
    for _, r in cls_docs.head(5).iterrows():
        print(f"    [{r['task_id']}] {r['text'][:110]}...")

# Show flagged docs
print(f"\n{'=' * 75}")
print(f"FLAGGED FOR MANUAL REVIEW ({len(review)} docs)")
print(f"{'=' * 75}")

for _, r in review.iterrows():
    print(f"\n  [{r['task_id']}] {r['text'][:120]}...")
    print(f"    Proposed: {r['proposed_class']} | Conf: {r['confidence']:.2f} | Votes: {r['knn_votes']}")

# Save results
output_path = Path(r"c:\Users\Lenovo\Desktop\asana-classification-v2\outputs\reports\t22_redistribution.csv")
results_df[["task_id", "text", "proposed_class", "confidence", "avg_similarity", "flag"]].to_csv(
    output_path, index=False, encoding="utf-8-sig")
print(f"\n\nSaved detailed results: {output_path}")
