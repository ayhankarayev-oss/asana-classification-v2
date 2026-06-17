"""
Phase 2 Audit: Comprehensive BERTopic Cluster Analysis
=======================================================
1. Distribution Audit (Topic_ID, Count, %, Name)
2. Probability Report (avg prob per topic, flag low confidence)
3. Similarity Report (topic pairs with similarity > 0.80)
4. Keyword Export (top 10 keywords per topic)
5. Optimization (Optuna) - tune min_cluster_size & min_samples
6. Pruning Proposal (noise clusters to merge/discard)
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===========================================================================
# Load data and rebuild model
# ===========================================================================
INPUT_CORPUS = Path("outputs/corpus_clean.csv")
EMBEDDINGS_CACHE = Path("outputs/embeddings.npy")

print("=" * 80)
print("PHASE 2 AUDIT: BERTopic Cluster Analysis")
print("=" * 80)

print("\n[LOADING DATA]")
df = pd.read_csv(INPUT_CORPUS, dtype=str, keep_default_na=False)
texts = df["cleaned_text"].tolist()
task_ids = df["task_id"].tolist()
embeddings = np.load(str(EMBEDDINGS_CACHE))
print(f"  Corpus: {len(texts):,} texts")
print(f"  Embeddings: {embeddings.shape}")

print("\n[REBUILDING MODEL] min_cluster_size=12, min_samples=1")
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=12, min_samples=1, metric="euclidean",
                         prediction_data=True, cluster_selection_method="eom")
vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    calculate_probabilities=True,
    verbose=False,
)
topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)
topic_info = topic_model.get_topic_info()

# Soft-clustering: reassign outliers
topics_arr = np.array(topics)
if probs is not None:
    for i in range(len(topics_arr)):
        if topics_arr[i] == -1 and len(probs.shape) == 2 and probs.shape[1] > 0:
            best_idx = np.argmax(probs[i])
            if probs[i][best_idx] > 0.0:
                topics_arr[i] = best_idx
topics = topics_arr.tolist()

print(f"  Topics: {len(topic_info[topic_info['Topic'] != -1])}")
print(f"  Final outliers: {sum(1 for t in topics if t == -1)}")

# ===========================================================================
# 1. DISTRIBUTION AUDIT
# ===========================================================================
print("\n" + "=" * 80)
print("1. DISTRIBUTION AUDIT")
print("=" * 80)

dist_data = []
for _, row in topic_info[topic_info["Topic"] != -1].iterrows():
    tid = row["Topic"]
    count = sum(1 for t in topics if t == tid)
    pct = count / len(topics) * 100
    topic_words = topic_model.get_topic(tid)
    name = "_".join([w for w, _ in topic_words[:4]]) if topic_words else f"topic_{tid}"
    dist_data.append({"Topic_ID": tid, "Count": count, "Pct": pct, "Name": name})

dist_df = pd.DataFrame(dist_data).sort_values("Count", ascending=False).reset_index(drop=True)
print(f"\n{'Topic_ID':<10}{'Count':<8}{'%':<8}{'Name'}")
print("-" * 80)
for _, row in dist_df.iterrows():
    print(f"{row['Topic_ID']:<10}{row['Count']:<8}{row['Pct']:<8.1f}{row['Name']}")

# ===========================================================================
# 2. PROBABILITY REPORT
# ===========================================================================
print("\n" + "=" * 80)
print("2. PROBABILITY REPORT")
print("=" * 80)

prob_data = []
if probs is not None and len(probs.shape) == 2:
    for tid in sorted(set(topics)):
        if tid == -1:
            continue
        # Get indices of docs assigned to this topic
        indices = [i for i, t in enumerate(topics) if t == tid]
        if indices:
            # Max probability for each doc assigned to this topic
            avg_prob = np.mean([probs[i].max() for i in indices])
            prob_data.append({
                "Topic_ID": tid,
                "Avg_Probability": avg_prob,
                "Flag": "LOW CONFIDENCE" if avg_prob < 0.3 else "OK"
            })

prob_df = pd.DataFrame(prob_data).sort_values("Avg_Probability")
low_conf = prob_df[prob_df["Flag"] == "LOW CONFIDENCE"]
print(f"\n  Total topics: {len(prob_df)}")
print(f"  Low Confidence (avg prob < 0.3): {len(low_conf)} topics")

if len(low_conf) > 0:
    print(f"\n  {'Topic_ID':<10}{'Avg_Prob':<12}{'Flag'}")
    print("  " + "-" * 40)
    for _, row in low_conf.iterrows():
        topic_words = topic_model.get_topic(row['Topic_ID'])
        name = "_".join([w for w, _ in topic_words[:3]]) if topic_words else ""
        print(f"  {row['Topic_ID']:<10}{row['Avg_Probability']:<12.4f}{row['Flag']}  ({name})")
else:
    print("  All topics have acceptable confidence (>= 0.3)")

# Show all topics with their probabilities
print(f"\n  Full probability table:")
print(f"  {'Topic_ID':<10}{'Avg_Prob':<12}{'Flag':<16}{'Name'}")
print("  " + "-" * 70)
for _, row in prob_df.iterrows():
    tid = row['Topic_ID']
    topic_words = topic_model.get_topic(tid)
    name = "_".join([w for w, _ in topic_words[:3]]) if topic_words else ""
    print(f"  {tid:<10}{row['Avg_Probability']:<12.4f}{row['Flag']:<16}{name}")

# ===========================================================================
# 3. SIMILARITY REPORT
# ===========================================================================
print("\n" + "=" * 80)
print("3. SIMILARITY REPORT (Topic pairs with similarity > 0.80)")
print("=" * 80)

# Get topic embeddings (c-TF-IDF based)
try:
    topic_embeddings = topic_model.topic_embeddings_
    if topic_embeddings is not None:
        # topic_embeddings[0] is the outlier topic (-1), rest are topics 0, 1, 2...
        # Remove outlier embedding
        valid_topics = sorted([t for t in set(topics) if t != -1])
        # Build similarity matrix between topic embeddings
        # topic_embeddings indices: 0=outlier(-1), 1=topic_0, 2=topic_1, ...
        topic_embs = topic_embeddings[1:]  # Skip outlier
        sim_matrix = cosine_similarity(topic_embs)
        
        # Find pairs with similarity > 0.80
        similar_pairs = []
        for i in range(len(valid_topics)):
            for j in range(i + 1, len(valid_topics)):
                sim = sim_matrix[i][j]
                if sim > 0.80:
                    similar_pairs.append({
                        "Topic_A": valid_topics[i],
                        "Topic_B": valid_topics[j],
                        "Similarity": sim,
                    })
        
        similar_pairs.sort(key=lambda x: -x["Similarity"])
        print(f"\n  Topic pairs with similarity > 0.80: {len(similar_pairs)}")
        
        if similar_pairs:
            print(f"\n  {'Topic_A':<10}{'Topic_B':<10}{'Similarity':<12}{'Names'}")
            print("  " + "-" * 70)
            for pair in similar_pairs[:30]:  # Show top 30
                ta, tb = pair["Topic_A"], pair["Topic_B"]
                words_a = topic_model.get_topic(ta)
                words_b = topic_model.get_topic(tb)
                name_a = "_".join([w for w, _ in words_a[:2]]) if words_a else ""
                name_b = "_".join([w for w, _ in words_b[:2]]) if words_b else ""
                print(f"  {ta:<10}{tb:<10}{pair['Similarity']:<12.4f}{name_a} <-> {name_b}")
        else:
            print("  No topic pairs found with similarity > 0.80")
    else:
        print("  Topic embeddings not available")
except Exception as e:
    print(f"  Error computing similarity: {e}")

# ===========================================================================
# 4. KEYWORD EXPORT
# ===========================================================================
print("\n" + "=" * 80)
print("4. KEYWORD EXPORT (Top 10 keywords per topic)")
print("=" * 80)

keyword_data = []
for tid in sorted(set(topics)):
    if tid == -1:
        continue
    topic_words = topic_model.get_topic(tid)
    if topic_words:
        keywords = [w for w, _ in topic_words[:10]]
        keyword_data.append({
            "Topic_ID": tid,
            "Keywords": ", ".join(keywords)
        })

keyword_df = pd.DataFrame(keyword_data)
print(f"\n  Exporting keywords for {len(keyword_df)} topics:\n")
print(f"  {'Topic_ID':<10}{'Keywords'}")
print("  " + "-" * 70)
for _, row in keyword_df.iterrows():
    print(f"  {row['Topic_ID']:<10}{row['Keywords'][:70]}")

# Save to CSV
keyword_df.to_csv("outputs/topic_keywords.csv", index=False, encoding="utf-8-sig")
print(f"\n  Saved: outputs/topic_keywords.csv")

# ===========================================================================
# 5. OPTUNA OPTIMIZATION
# ===========================================================================
print("\n" + "=" * 80)
print("5. OPTUNA OPTIMIZATION")
print("=" * 80)
print("  Optimizing min_cluster_size (5-20) and min_samples (1-5)")
print("  Objective: maximize coherence while keeping outliers < 10%")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Pre-compute UMAP reduction (shared across all trials)
print("  Pre-computing UMAP reduction...")
umap_reduced = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
reduced_embeddings = umap_reduced.fit_transform(embeddings)


def objective(trial):
    """Optuna objective: maximize coherence with outlier constraint."""
    mcs = trial.suggest_int("min_cluster_size", 5, 20)
    ms = trial.suggest_int("min_samples", 1, 5)
    
    # Cluster with HDBSCAN
    hdbscan_trial = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = hdbscan_trial.fit_predict(reduced_embeddings)
    
    # Outlier rate
    n_outliers = (labels == -1).sum()
    outlier_rate = n_outliers / len(labels)
    
    # Penalty: if outliers > 10%, penalize heavily
    if outlier_rate > 0.10:
        return -1.0 + (0.10 - outlier_rate)  # Negative, worse the higher outliers are
    
    # Number of non-outlier clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 5:
        return -2.0  # Too few clusters
    
    # Coherence proxy: average cluster compactness (within-cluster cosine similarity)
    coherence_scores = []
    for c in set(labels):
        if c == -1:
            continue
        mask = labels == c
        if mask.sum() < 2:
            continue
        cluster_embs = embeddings[mask]
        # Average pairwise cosine similarity within cluster
        if len(cluster_embs) > 50:
            # Sample for speed
            idx = np.random.choice(len(cluster_embs), 50, replace=False)
            cluster_embs = cluster_embs[idx]
        sim = cosine_similarity(cluster_embs)
        np.fill_diagonal(sim, 0)
        avg_sim = sim.sum() / (len(sim) * (len(sim) - 1))
        coherence_scores.append(avg_sim)
    
    if not coherence_scores:
        return -2.0
    
    # Combine: high coherence + reasonable number of topics (20-80)
    avg_coherence = np.mean(coherence_scores)
    topic_penalty = 0
    if n_clusters > 80:
        topic_penalty = (n_clusters - 80) * 0.005
    elif n_clusters < 20:
        topic_penalty = (20 - n_clusters) * 0.01
    
    return avg_coherence - topic_penalty


print("  Running 50 trials...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=False)

best = study.best_params
best_val = study.best_value
print(f"\n  Best parameters:")
print(f"    min_cluster_size = {best['min_cluster_size']}")
print(f"    min_samples      = {best['min_samples']}")
print(f"    Coherence score  = {best_val:.4f}")

# Show what the best config produces
hdbscan_best = HDBSCAN(
    min_cluster_size=best['min_cluster_size'],
    min_samples=best['min_samples'],
    metric="euclidean",
    cluster_selection_method="eom",
)
best_labels = hdbscan_best.fit_predict(reduced_embeddings)
n_outliers_best = (best_labels == -1).sum()
n_clusters_best = len(set(best_labels)) - (1 if -1 in best_labels else 0)
print(f"    Topics produced  = {n_clusters_best}")
print(f"    Outlier rate     = {n_outliers_best/len(best_labels)*100:.1f}%")

# ===========================================================================
# 6. PRUNING PROPOSAL
# ===========================================================================
print("\n" + "=" * 80)
print("6. PRUNING PROPOSAL")
print("=" * 80)

noise_clusters = []
merge_candidates = []

# Flag low-confidence topics
for _, row in prob_df[prob_df["Flag"] == "LOW CONFIDENCE"].iterrows():
    tid = int(row["Topic_ID"])
    count = sum(1 for t in topics if t == tid)
    topic_words = topic_model.get_topic(tid)
    name = "_".join([w for w, _ in topic_words[:3]]) if topic_words else ""
    noise_clusters.append({
        "Topic_ID": tid,
        "Count": count,
        "Reason": f"Low confidence (avg_prob={row['Avg_Probability']:.3f})",
        "Name": name,
    })

# Flag very small clusters (< 15 docs, likely noise from soft-assignment)
for tid in sorted(set(topics)):
    if tid == -1:
        continue
    count = sum(1 for t in topics if t == tid)
    if count < 15:
        already_flagged = any(n["Topic_ID"] == tid for n in noise_clusters)
        if not already_flagged:
            topic_words = topic_model.get_topic(tid)
            name = "_".join([w for w, _ in topic_words[:3]]) if topic_words else ""
            noise_clusters.append({
                "Topic_ID": tid,
                "Count": count,
                "Reason": f"Very small cluster ({count} docs)",
                "Name": name,
            })

# Flag similar pairs for merging
if similar_pairs:
    for pair in similar_pairs:
        ta, tb = pair["Topic_A"], pair["Topic_B"]
        words_a = topic_model.get_topic(ta)
        words_b = topic_model.get_topic(tb)
        name_a = "_".join([w for w, _ in words_a[:3]]) if words_a else ""
        name_b = "_".join([w for w, _ in words_b[:3]]) if words_b else ""
        merge_candidates.append({
            "Topic_A": ta,
            "Topic_B": tb,
            "Similarity": pair["Similarity"],
            "Suggestion": f"Merge '{name_a}' + '{name_b}'"
        })

print(f"\n  A. NOISE CLUSTERS (suggest discard/reassign): {len(noise_clusters)}")
if noise_clusters:
    print(f"\n  {'Topic_ID':<10}{'Count':<8}{'Reason':<40}{'Name'}")
    print("  " + "-" * 70)
    for nc in sorted(noise_clusters, key=lambda x: x["Count"]):
        print(f"  {nc['Topic_ID']:<10}{nc['Count']:<8}{nc['Reason']:<40}{nc['Name']}")

print(f"\n  B. MERGE CANDIDATES (similarity > 0.80): {len(merge_candidates)}")
if merge_candidates:
    print(f"\n  {'Topic_A':<10}{'Topic_B':<10}{'Sim':<10}{'Suggestion'}")
    print("  " + "-" * 70)
    for mc in merge_candidates:
        print(f"  {mc['Topic_A']:<10}{mc['Topic_B']:<10}{mc['Similarity']:<10.4f}{mc['Suggestion']}")
else:
    print("  No merge candidates (all topic pairs have similarity <= 0.80)")

# Summary
total_noise_docs = sum(nc["Count"] for nc in noise_clusters)
print(f"\n  SUMMARY:")
print(f"    Noise clusters to prune: {len(noise_clusters)} ({total_noise_docs} docs, {total_noise_docs/len(topics)*100:.1f}%)")
print(f"    Merge candidates: {len(merge_candidates)} pairs")
print(f"    Clean topics (keep as-is): {len(set(topics)) - 1 - len(noise_clusters)} topics")

# Save full audit
dist_df.to_csv("outputs/audit_distribution.csv", index=False, encoding="utf-8-sig")
prob_df.to_csv("outputs/audit_probabilities.csv", index=False, encoding="utf-8-sig")
keyword_df.to_csv("outputs/topic_keywords.csv", index=False, encoding="utf-8-sig")
print(f"\n  Saved: outputs/audit_distribution.csv")
print(f"  Saved: outputs/audit_probabilities.csv")
print(f"  Saved: outputs/topic_keywords.csv")

print(f"\n{'=' * 80}")
print("AUDIT COMPLETE")
print(f"{'=' * 80}")
