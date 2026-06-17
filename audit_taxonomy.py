"""
Cluster Audit & Taxonomy Report Generator
==========================================
Generates a persistent, comprehensive report of all BERTopic clusters.
Re-run this script any time the model changes to update the report.

Outputs (all saved to outputs/reports/):
  1. cluster_audit_report.html   - Full interactive HTML report
  2. taxonomy_draft.csv          - Draft taxonomy mapping (for Phase 5 input)
  3. merge_candidates.csv        - Pairs with similarity > 0.75
  4. cross_tabulation.csv        - Topic x Platform frequency table
  5. cluster_samples.csv         - 50 sample docs per cluster with GIDs
  6. split_analysis.csv          - Split detection results per cluster

Usage:
  python audit_taxonomy.py

The HTML report is self-contained and can be shared for review.
"""
import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

# ===========================================================================
# Config
# ===========================================================================
INPUT_CORPUS = Path("outputs/corpus_clean.csv")
EMBEDDINGS_CACHE = Path("outputs/embeddings.npy")
REPORT_DIR = Path("outputs/reports")
SAMPLES_PER_TOPIC = 50
SIMILARITY_THRESHOLD = 0.75
SPLIT_SILHOUETTE_THRESHOLD = 0.15
MISCLASS_PERCENTILE = 5  # Bottom 5% flagged

# Medium resolution params
MIN_CLUSTER_SIZE = 30
MIN_SAMPLES = 3


def load_data():
    """Load corpus and embeddings."""
    df = pd.read_csv(INPUT_CORPUS, dtype=str, keep_default_na=False)
    embeddings = np.load(str(EMBEDDINGS_CACHE))
    return df, embeddings


def build_model(texts, embeddings):
    """Build and fit BERTopic model."""
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES,
                            metric="euclidean", prediction_data=True,
                            cluster_selection_method="eom")
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=True,
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)

    # Soft-clustering reassignment
    topics_arr = np.array(topics)
    if probs is not None and len(probs.shape) == 2:
        for i in range(len(topics_arr)):
            if topics_arr[i] == -1 and probs.shape[1] > 0:
                best_idx = np.argmax(probs[i])
                if probs[i][best_idx] > 0.0:
                    topics_arr[i] = best_idx
    return topic_model, topics_arr.tolist(), probs


def compute_split_analysis(topic_id, topic_indices, embeddings, texts):
    """Detect if a topic should be split into 2 sub-themes."""
    if len(topic_indices) < 30:
        return {"split_recommended": False, "silhouette": 0, "reason": "Too few docs"}

    topic_embs = embeddings[topic_indices]
    topic_texts = [texts[i] for i in topic_indices]

    # KMeans with k=2
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    sub_labels = km.fit_predict(topic_embs)

    # Silhouette score
    sil = silhouette_score(topic_embs, sub_labels, metric="cosine")

    # Get sub-theme keywords
    sub_keywords = {}
    for sub_id in [0, 1]:
        sub_texts = [topic_texts[i] for i in range(len(topic_texts)) if sub_labels[i] == sub_id]
        if len(sub_texts) >= 5:
            vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5, min_df=2)
            try:
                vec.fit(sub_texts)
                sub_keywords[sub_id] = list(vec.vocabulary_.keys())[:5]
            except ValueError:
                sub_keywords[sub_id] = []
        else:
            sub_keywords[sub_id] = []

    count_a = int((sub_labels == 0).sum())
    count_b = int((sub_labels == 1).sum())

    return {
        "split_recommended": sil > SPLIT_SILHOUETTE_THRESHOLD,
        "silhouette": float(sil),
        "sub_a_count": count_a,
        "sub_b_count": count_b,
        "sub_a_keywords": sub_keywords.get(0, []),
        "sub_b_keywords": sub_keywords.get(1, []),
        "reason": f"Silhouette={sil:.3f} {'> threshold' if sil > SPLIT_SILHOUETTE_THRESHOLD else '< threshold'}"
    }


def detect_misclassifications(topic_indices, embeddings, n_flag=5):
    """Flag docs furthest from cluster centroid."""
    if len(topic_indices) < 10:
        return []

    topic_embs = embeddings[topic_indices]
    centroid = topic_embs.mean(axis=0, keepdims=True)
    sims = cosine_similarity(topic_embs, centroid).flatten()

    # Bottom N by similarity
    n_flag = max(1, min(n_flag, int(len(topic_indices) * MISCLASS_PERCENTILE / 100)))
    worst_indices = np.argsort(sims)[:n_flag]

    results = []
    for idx in worst_indices:
        results.append({
            "local_idx": int(idx),
            "global_idx": int(topic_indices[idx]),
            "similarity_to_centroid": float(sims[idx])
        })
    return results


def generate_html_report(report_data, timestamp):
    """Generate a self-contained HTML report."""
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cluster Audit Report - {timestamp}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #f8f9fa; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
h2 {{ color: #16213e; margin-top: 40px; }}
h3 {{ color: #0f3460; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }}
th {{ background: #16213e; color: white; padding: 10px 8px; text-align: left; }}
td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
.cluster-card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
.badge-ok {{ background: #d4edda; color: #155724; }}
.badge-warn {{ background: #fff3cd; color: #856404; }}
.badge-danger {{ background: #f8d7da; color: #721c24; }}
.badge-info {{ background: #d1ecf1; color: #0c5460; }}
.sample {{ font-size: 12px; color: #555; margin: 2px 0; font-family: monospace; }}
.meta {{ color: #666; font-size: 12px; margin-bottom: 10px; }}
.summary-box {{ background: #e8f4f8; border-left: 4px solid #17a2b8; padding: 15px; margin: 20px 0; }}
</style>
</head>
<body>
<h1>Cluster Audit Report</h1>
<div class="meta">Generated: {timestamp} | Model: BERTopic (min_cluster_size={MIN_CLUSTER_SIZE}) | Corpus: {report_data['total_docs']} docs</div>

<div class="summary-box">
<strong>Summary:</strong> {report_data['n_topics']} topics | 0% outliers | {report_data['n_merge_candidates']} merge candidates | {report_data['n_splits_recommended']} split recommendations
</div>

<h2>1. Distribution Overview</h2>
<table>
<tr><th>Topic</th><th>Suggested Name</th><th>Count</th><th>%</th><th>Coherence</th><th>Confidence</th><th>Split?</th><th>Merge?</th></tr>
"""

    for t in report_data["topics"]:
        conf_badge = "badge-danger" if t["confidence"] == "LOW" else "badge-ok" if t["confidence"] == "HIGH" else "badge-warn"
        split_badge = "badge-warn" if t["split_recommended"] else "badge-ok"
        merge_text = ", ".join([str(m) for m in t["merge_with"]]) if t["merge_with"] else "-"
        html += f"""<tr>
<td><strong>{t['topic_id']}</strong></td>
<td>{t['suggested_name']}</td>
<td>{t['count']}</td>
<td>{t['pct']:.1f}%</td>
<td>{t['coherence']:.3f}</td>
<td><span class="badge {conf_badge}">{t['confidence']}</span></td>
<td><span class="badge {split_badge}">{'SPLIT' if t['split_recommended'] else 'OK'}</span></td>
<td>{merge_text}</td>
</tr>"""

    html += "</table>"

    # Cross-tabulation
    html += "<h2>2. Platform Cross-Tabulation</h2><table><tr><th>Topic</th><th>Name</th>"
    platforms = report_data["platforms"]
    for p in platforms:
        html += f"<th>{p}</th>"
    html += "</tr>"
    for t in report_data["topics"]:
        html += f"<tr><td>{t['topic_id']}</td><td>{t['suggested_name'][:30]}</td>"
        for p in platforms:
            val = t["platform_dist"].get(p, 0)
            html += f"<td>{val}</td>"
        html += "</tr>"
    html += "</table>"

    # Merge candidates
    html += "<h2>3. Merge Candidates (Similarity > 0.75)</h2><table><tr><th>Topic A</th><th>Topic B</th><th>Similarity</th><th>Keywords A</th><th>Keywords B</th><th>Recommendation</th></tr>"
    for m in report_data["merge_candidates"]:
        rec_badge = "badge-danger" if m["similarity"] > 0.85 else "badge-warn"
        html += f"""<tr><td>{m['topic_a']}</td><td>{m['topic_b']}</td><td>{m['similarity']:.4f}</td>
<td>{m['keywords_a']}</td><td>{m['keywords_b']}</td>
<td><span class="badge {rec_badge}">{m['recommendation']}</span></td></tr>"""
    html += "</table>"

    # Per-cluster detail
    html += "<h2>4. Per-Cluster Detail</h2>"
    for t in report_data["topics"]:
        split_info = ""
        if t["split_recommended"]:
            split_info = f"""<p><strong>Split Recommended</strong> (silhouette={t['split_silhouette']:.3f})</p>
<p>Sub-theme A ({t['split_a_count']} docs): {', '.join(t['split_a_kw'])}</p>
<p>Sub-theme B ({t['split_b_count']} docs): {', '.join(t['split_b_kw'])}</p>"""

        misclass_html = ""
        if t["misclassifications"]:
            misclass_html = "<p><strong>Potential Misclassifications:</strong></p>"
            for mc in t["misclassifications"]:
                misclass_html += f"<div class='sample'>[{mc['task_id']}] sim={mc['similarity']:.3f} | {mc['text'][:100]}...</div>"

        samples_html = ""
        for s in t["samples"][:50]:
            samples_html += f"<div class='sample'>[{s['task_id']}] {s['text'][:120]}</div>"

        html += f"""<div class="cluster-card">
<h3>Topic {t['topic_id']}: {t['suggested_name']}</h3>
<p><strong>Keywords:</strong> {t['keywords_full']}</p>
<p><strong>Docs:</strong> {t['count']} | <strong>Coherence:</strong> {t['coherence']:.3f} | <strong>Confidence:</strong> {t['confidence']}</p>
<p><strong>Platforms:</strong> {json.dumps(t['platform_dist'])}</p>
{split_info}
{misclass_html}
<details><summary>Samples ({min(50, t['count'])} docs)</summary>{samples_html}</details>
</div>"""

    html += f"""
<hr>
<p class="meta">Report auto-generated by audit_taxonomy.py | Last updated: {timestamp}</p>
</body></html>"""
    return html


def suggest_name(topic_id, keywords, platform_dist):
    """Heuristic business name suggestion based on keywords and platform."""
    kw = [k.lower() for k in keywords[:6]]
    kw_str = " ".join(kw)

    # Platform-specific naming
    dominant_platform = max(platform_dist, key=platform_dist.get) if platform_dist else "General"
    platform_pct = platform_dist.get(dominant_platform, 0) / max(sum(platform_dist.values()), 1)

    # Keyword-based rules
    if "new account" in kw_str or ("account" in kw and "new" in kw):
        return "New Account Setup"
    if "private investment" in kw_str or "new private" in kw_str:
        return "New Private Investment"
    if "k1" in kw or "lp" in kw:
        return "Fund Administration (K-1/LP)"
    if "portal" in kw or "view" in kw:
        return "Portal & Report Views"
    if "excel" in kw or "commitment" in kw or "unfunded" in kw:
        return "Commitment & Excel Data"
    if "access" in kw or "yes yes" in kw_str:
        return "Access & Permissions"
    if "meeting" in kw or "reminder" in kw or "schedule" in kw:
        return "Meetings & Follow-ups"
    if "update person" in kw_str or ("person" in kw and "update" in kw):
        return "Contact/Person Updates"
    if "ownership" in kw or "trust" in kw or "structure" in kw:
        return "Ownership & Trust Structure"
    if "asset" in kw and "class" in kw:
        return "Asset Classification"
    if "invoice" in kw or "invoices" in kw:
        return "Invoicing & Billing"
    if "valuation" in kw and "import" in kw:
        name = "Valuation Import"
        if platform_pct > 0.6 and dominant_platform != "General":
            name += f" ({dominant_platform})"
        return name
    if "distribution" in kw or "transaction type" in kw_str:
        return "Distributions & Transactions"
    if "performance" in kw or "benchmark" in kw:
        return "Performance & Benchmarks"
    if "loan" in kw or "promissory" in kw:
        return "Loans & Promissory Notes"
    if "backfill" in kw or ("data" in kw and "statements" in kw):
        return "Data Backfill & Statements"
    if "missing" in kw and "data" in kw:
        return "Missing Data Resolution"
    if "cost basis" in kw_str or "cost" in kw:
        return "Cost Basis Issues"
    if "transfer" in kw:
        return "Transfers & General Updates"
    if "owner" in kw or "direct owner" in kw_str:
        return "Direct Owner & Asset Views"
    if "id" in kw and "investors" in kw:
        return "Investor Entity Management"
    if "llc" in kw and "series" in kw:
        return "LLC/Series Entity Setup"

    return f"Topic_{topic_id}"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print(f"CLUSTER AUDIT & TAXONOMY REPORT GENERATOR")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)

    # Load
    print("\n[1/6] Loading data...")
    df, embeddings = load_data()
    texts = df["cleaned_text"].tolist()
    task_ids = df["task_id"].tolist()
    print(f"  Corpus: {len(df)} docs | Embeddings: {embeddings.shape}")

    # Build model
    print("\n[2/6] Building BERTopic model (medium resolution)...")
    topic_model, topics, probs = build_model(texts, embeddings)
    df["topic_id"] = topics
    valid_topics = sorted(set(t for t in topics if t != -1))
    print(f"  Topics: {len(valid_topics)} | Outliers: {sum(1 for t in topics if t == -1)}")

    # Analyze each cluster
    print("\n[3/6] Analyzing clusters...")
    report_topics = []
    all_samples = []
    split_results = []

    for tid in valid_topics:
        topic_indices = [i for i, t in enumerate(topics) if t == tid]
        count = len(topic_indices)
        pct = count / len(topics) * 100

        # Keywords
        words = topic_model.get_topic(tid)
        keywords = [w for w, _ in words[:10]] if words else []
        keywords_full = ", ".join(keywords)

        # Platform distribution
        topic_df = df[df["topic_id"] == tid]
        platform_dist = topic_df["primary_platform"].value_counts().to_dict()

        # Coherence (avg pairwise cosine sim within cluster, sampled)
        topic_embs = embeddings[topic_indices]
        if len(topic_embs) > 100:
            sample_idx = np.random.choice(len(topic_embs), 100, replace=False)
            sample_embs = topic_embs[sample_idx]
        else:
            sample_embs = topic_embs
        sim_matrix = cosine_similarity(sample_embs)
        np.fill_diagonal(sim_matrix, 0)
        coherence = sim_matrix.sum() / (len(sim_matrix) * (len(sim_matrix) - 1)) if len(sim_matrix) > 1 else 0

        # Confidence
        if probs is not None and len(probs.shape) == 2:
            avg_prob = np.mean([probs[i].max() for i in topic_indices])
        else:
            avg_prob = 0
        confidence = "LOW" if avg_prob < 0.3 else "HIGH" if avg_prob > 0.5 else "OK"

        # Split analysis
        split = compute_split_analysis(tid, topic_indices, embeddings, texts)
        split_results.append({"topic_id": tid, **split})

        # Misclassification detection
        misclass = detect_misclassifications(topic_indices, embeddings)
        misclass_with_text = []
        for mc in misclass:
            gidx = mc["global_idx"]
            misclass_with_text.append({
                "task_id": task_ids[gidx],
                "similarity": mc["similarity_to_centroid"],
                "text": texts[gidx][:120]
            })

        # Samples (50 docs)
        sample_indices = topic_indices[:SAMPLES_PER_TOPIC]
        samples = []
        for idx in sample_indices:
            samples.append({"task_id": task_ids[idx], "text": texts[idx][:150]})
            all_samples.append({"topic_id": tid, "task_id": task_ids[idx], "text": texts[idx][:200]})

        # Suggested name
        suggested_name = suggest_name(tid, keywords, platform_dist)

        report_topics.append({
            "topic_id": tid,
            "count": count,
            "pct": pct,
            "keywords": keywords,
            "keywords_full": keywords_full,
            "platform_dist": platform_dist,
            "coherence": coherence,
            "confidence": confidence,
            "avg_probability": avg_prob,
            "suggested_name": suggested_name,
            "split_recommended": split["split_recommended"],
            "split_silhouette": split["silhouette"],
            "split_a_count": split.get("sub_a_count", 0),
            "split_b_count": split.get("sub_b_count", 0),
            "split_a_kw": split.get("sub_a_keywords", []),
            "split_b_kw": split.get("sub_b_keywords", []),
            "misclassifications": misclass_with_text,
            "samples": samples,
            "merge_with": [],
        })

    # Similarity / Merge candidates
    print("\n[4/6] Computing merge candidates...")
    merge_candidates = []
    try:
        topic_embs_all = topic_model.topic_embeddings_[1:]
        sim_matrix_topics = cosine_similarity(topic_embs_all)
        for i in range(len(valid_topics)):
            for j in range(i + 1, len(valid_topics)):
                sim = sim_matrix_topics[i][j]
                if sim > SIMILARITY_THRESHOLD:
                    ta, tb = valid_topics[i], valid_topics[j]
                    words_a = topic_model.get_topic(ta)
                    words_b = topic_model.get_topic(tb)
                    kw_a = ", ".join([w for w, _ in words_a[:4]]) if words_a else ""
                    kw_b = ", ".join([w for w, _ in words_b[:4]]) if words_b else ""
                    rec = "MERGE" if sim > 0.85 else "CONSIDER"
                    merge_candidates.append({
                        "topic_a": ta, "topic_b": tb, "similarity": sim,
                        "keywords_a": kw_a, "keywords_b": kw_b, "recommendation": rec
                    })
                    # Tag the topics
                    for rt in report_topics:
                        if rt["topic_id"] == ta:
                            rt["merge_with"].append(tb)
                        if rt["topic_id"] == tb:
                            rt["merge_with"].append(ta)
    except Exception as e:
        print(f"  Warning: {e}")

    merge_candidates.sort(key=lambda x: -x["similarity"])
    print(f"  Found {len(merge_candidates)} merge candidates")

    # Generate outputs
    print("\n[5/6] Generating report files...")

    # Report data structure
    platforms = sorted(set(df["primary_platform"].unique()))
    report_data = {
        "timestamp": timestamp,
        "total_docs": len(df),
        "n_topics": len(valid_topics),
        "n_merge_candidates": len(merge_candidates),
        "n_splits_recommended": sum(1 for t in report_topics if t["split_recommended"]),
        "topics": report_topics,
        "merge_candidates": merge_candidates,
        "platforms": platforms,
    }

    # 1. HTML Report
    html = generate_html_report(report_data, timestamp)
    html_path = REPORT_DIR / "cluster_audit_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved: {html_path}")

    # 2. Taxonomy draft CSV
    tax_rows = []
    for t in report_topics:
        tax_rows.append({
            "topic_id": t["topic_id"],
            "current_label": t["keywords_full"][:50],
            "suggested_name": t["suggested_name"],
            "doc_count": t["count"],
            "pct": round(t["pct"], 1),
            "coherence": round(t["coherence"], 3),
            "confidence": t["confidence"],
            "avg_probability": round(t["avg_probability"], 4),
            "split_recommended": t["split_recommended"],
            "split_silhouette": round(t["split_silhouette"], 3),
            "merge_with": "|".join([str(m) for m in t["merge_with"]]) if t["merge_with"] else "",
        })
    tax_df = pd.DataFrame(tax_rows)
    tax_path = REPORT_DIR / "taxonomy_draft.csv"
    tax_df.to_csv(tax_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {tax_path}")

    # 3. Merge candidates CSV
    merge_df = pd.DataFrame(merge_candidates)
    merge_path = REPORT_DIR / "merge_candidates.csv"
    merge_df.to_csv(merge_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {merge_path}")

    # 4. Cross-tabulation CSV
    cross = pd.crosstab(df["topic_id"].astype(int), df["primary_platform"])
    cross_path = REPORT_DIR / "cross_tabulation.csv"
    cross.to_csv(cross_path, encoding="utf-8-sig")
    print(f"  Saved: {cross_path}")

    # 5. Cluster samples CSV
    samples_df = pd.DataFrame(all_samples)
    samples_path = REPORT_DIR / "cluster_samples.csv"
    samples_df.to_csv(samples_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {samples_path}")

    # 6. Split analysis CSV
    split_df = pd.DataFrame(split_results)
    split_path = REPORT_DIR / "split_analysis.csv"
    split_df.to_csv(split_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {split_path}")

    # Summary
    print(f"\n[6/6] Report generation complete!")
    print(f"\n{'=' * 70}")
    print(f"REPORT FILES (outputs/reports/):")
    print(f"{'=' * 70}")
    print(f"  1. cluster_audit_report.html  - Full interactive HTML report")
    print(f"  2. taxonomy_draft.csv         - Draft taxonomy ({len(valid_topics)} topics)")
    print(f"  3. merge_candidates.csv       - {len(merge_candidates)} merge pairs")
    print(f"  4. cross_tabulation.csv       - Topic x Platform matrix")
    print(f"  5. cluster_samples.csv        - {len(all_samples)} sample docs")
    print(f"  6. split_analysis.csv         - Split detection results")
    print(f"\n  Open cluster_audit_report.html in a browser for the full report.")
    print(f"  Re-run this script any time to update with latest results.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
