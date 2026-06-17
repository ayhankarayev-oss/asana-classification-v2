"""
Phase 2: Multi-Resolution Discovery Scan
==========================================
Runs BERTopic unsupervised topic discovery at 3 cluster granularities.
Shared embeddings computed once, clustered at different resolutions.

Input:  outputs/corpus_clean.csv (from Phase 1)
Output: outputs/discovery_fine.csv     (min_cluster_size=20)
        outputs/discovery_medium.csv   (min_cluster_size=40)
        outputs/discovery_coarse.csv   (min_cluster_size=80)
        outputs/embeddings.npy         (cached embeddings for reuse)

Traceability:
    Every output row preserves task_id (Asana GID) from input.
    Script verifies zero task IDs lost during clustering.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# ===========================================================================
# Paths
# ===========================================================================
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

INPUT_CORPUS = Path("outputs/corpus_clean.csv")
OUTPUT_DIR = Path("outputs")
EMBEDDINGS_CACHE = OUTPUT_DIR / "embeddings.npy"

RESOLUTIONS = {
    "fine":   {"min_cluster_size": 12, "min_samples": 1},
    "medium": {"min_cluster_size": 30, "min_samples": 3},
    "coarse": {"min_cluster_size": 60, "min_samples": 5},
}

EMBEDDING_MODEL = "all-mpnet-base-v2"


def load_corpus() -> pd.DataFrame:
    """Load and validate input corpus."""
    if not INPUT_CORPUS.exists():
        sys.exit(f"ERROR: Input file not found: {INPUT_CORPUS}")

    df = pd.read_csv(INPUT_CORPUS, dtype=str, keep_default_na=False)

    # Validation
    assert "task_id" in df.columns, "Missing 'task_id' column"
    assert "cleaned_text" in df.columns, "Missing 'cleaned_text' column"
    assert df["task_id"].nunique() == len(df), "Duplicate task_ids in input"
    assert (df["task_id"] != "").all(), "Empty task_ids found"

    print(f"  Loaded: {len(df):,} rows")
    print(f"  task_id validation: PASS (all unique, no nulls)")
    return df


def compute_embeddings(texts: list[str]) -> np.ndarray:
    """Compute or load cached sentence embeddings."""
    if EMBEDDINGS_CACHE.exists():
        print(f"  Loading cached embeddings from {EMBEDDINGS_CACHE}")
        embeddings = np.load(str(EMBEDDINGS_CACHE))
        if len(embeddings) == len(texts):
            print(f"  Cached embeddings valid: shape {embeddings.shape}")
            return embeddings
        print(f"  Cache size mismatch ({len(embeddings)} vs {len(texts)}), recomputing...")

    print(f"  Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"  Encoding {len(texts):,} texts (this may take 2-3 minutes on CPU)...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )
    elapsed = time.time() - t0
    print(f"  Embeddings computed: shape {embeddings.shape} in {elapsed:.1f}s")

    # Cache for reuse
    np.save(str(EMBEDDINGS_CACHE), embeddings)
    print(f"  Cached to {EMBEDDINGS_CACHE}")

    return embeddings


def run_clustering(
    texts: list[str],
    embeddings: np.ndarray,
    task_ids: list[str],
    resolution_name: str,
    min_cluster_size: int,
    min_samples: int,
) -> pd.DataFrame:
    """
    Run BERTopic at a single resolution.
    Returns DataFrame with task_id, topic_id, topic_label, representative.
    """
    print(f"\n  [{resolution_name.upper()}] min_cluster_size={min_cluster_size}, min_samples={min_samples}")

    # Configure sub-models
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=True,
        cluster_selection_method="eom",
    )

    # Vectorizer for topic representations (use common English stop words)
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    # Build BERTopic model (calculate_probabilities=True for soft clustering)
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=True,
        verbose=False,
    )

    # Fit on pre-computed embeddings
    t0 = time.time()
    topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)
    elapsed = time.time() - t0

    # Get topic info
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info[topic_info["Topic"] != -1])
    n_outliers_raw = (np.array(topics) == -1).sum()

    print(f"    Topics found: {n_topics}")
    print(f"    Outliers (hard clustering): {n_outliers_raw:,} ({n_outliers_raw/len(texts)*100:.1f}%)")
    print(f"    Time: {elapsed:.1f}s")

    # Soft-clustering: reassign outliers to nearest topic using probabilities
    if probs is not None and n_outliers_raw > 0:
        topics_arr = np.array(topics)
        reassigned = 0
        for i in range(len(topics_arr)):
            if topics_arr[i] == -1:
                # Find the topic with highest probability for this outlier
                if len(probs.shape) == 2 and probs.shape[1] > 0:
                    best_topic_idx = np.argmax(probs[i])
                    best_prob = probs[i][best_topic_idx]
                    if best_prob > 0.0:
                        topics_arr[i] = best_topic_idx
                        reassigned += 1
        topics = topics_arr.tolist()
        n_outliers_final = sum(1 for t in topics if t == -1)
        print(f"    Soft-clustering reassigned: {reassigned:,} outliers -> nearest topic")
        print(f"    Final outliers: {n_outliers_final:,} ({n_outliers_final/len(texts)*100:.1f}%)")
    else:
        n_outliers_final = n_outliers_raw
        print(f"    Probabilities not available, no soft-clustering applied")

    # --- VALIDATION: Print outlier percentage ---
    outlier_pct = n_outliers_final / len(texts) * 100
    target_met = outlier_pct < 20
    print(f"\n    >>> OUTLIER RATE: {outlier_pct:.1f}% {'(TARGET MET: < 20%)' if target_met else '(ABOVE TARGET)'}")

    # Build topic label lookup
    topic_labels = {}
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            topic_labels[-1] = "outlier"
        else:
            # Use top keywords as label
            topic_words = topic_model.get_topic(tid)
            if topic_words:
                label = "_".join([w for w, _ in topic_words[:4]])
                topic_labels[tid] = label
            else:
                topic_labels[tid] = f"topic_{tid}"

    # Get representative docs for each topic
    representative_docs = {}
    try:
        rep_docs = topic_model.get_representative_docs()
        if rep_docs:
            for tid, docs in rep_docs.items():
                representative_docs[tid] = docs[0][:200] if docs else ""
    except Exception:
        pass

    # Build output DataFrame
    result_df = pd.DataFrame({
        "task_id": task_ids,
        "topic_id": topics,
        "topic_label": [topic_labels.get(t, f"topic_{t}") for t in topics],
        "representative_doc": [representative_docs.get(t, "") for t in topics],
    })

    # Print top topics
    print(f"    Top topics:")
    top_topics = topic_info[topic_info["Topic"] != -1].head(10)
    for _, row in top_topics.iterrows():
        tid = row["Topic"]
        count = row["Count"]
        label = topic_labels.get(tid, "")
        print(f"      Topic {tid:>3}: {count:>4} docs | {label}")

    return result_df, topic_model


def verify_traceability(input_df: pd.DataFrame, output_df: pd.DataFrame, resolution: str) -> bool:
    """Verify no task IDs were lost during clustering."""
    input_ids = set(input_df["task_id"].tolist())
    output_ids = set(output_df["task_id"].tolist())

    lost = input_ids - output_ids
    extra = output_ids - input_ids

    if lost:
        print(f"  ERROR [{resolution}]: {len(lost)} task_ids LOST!")
        return False
    if extra:
        print(f"  ERROR [{resolution}]: {len(extra)} extra task_ids appeared!")
        return False

    assert len(output_df) == len(input_df), \
        f"Row count mismatch: input={len(input_df)}, output={len(output_df)}"

    print(f"  [{resolution}] Traceability PASS: {len(output_df):,} task_ids preserved (0 lost)")
    return True


def main():
    print("=" * 60)
    print("PHASE 2: MULTI-RESOLUTION DISCOVERY SCAN")
    print("=" * 60)

    # --- Load ---
    print("\n[LOAD]")
    print(f"  Input: {INPUT_CORPUS.resolve()}")
    df = load_corpus()

    texts = df["cleaned_text"].tolist()
    task_ids = df["task_id"].tolist()

    # --- Embed ---
    print("\n[EMBEDDING]")
    embeddings = compute_embeddings(texts)

    # --- Cluster at each resolution ---
    print("\n[CLUSTERING]")
    all_passed = True

    for resolution_name, params in RESOLUTIONS.items():
        result_df, topic_model = run_clustering(
            texts=texts,
            embeddings=embeddings,
            task_ids=task_ids,
            resolution_name=resolution_name,
            **params,
        )

        # Save output
        output_path = OUTPUT_DIR / f"discovery_{resolution_name}.csv"
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"    Saved: {output_path}")

        # Save HTML visualization
        try:
            fig = topic_model.visualize_barchart(top_n_topics=15)
            html_path = OUTPUT_DIR / f"topics_{resolution_name}_barchart.html"
            fig.write_html(str(html_path))
            print(f"    Visualization: {html_path}")
        except Exception as e:
            print(f"    Visualization skipped: {e}")

    # --- Traceability verification ---
    print("\n[TRACEABILITY VERIFICATION]")
    for resolution_name in RESOLUTIONS:
        output_path = OUTPUT_DIR / f"discovery_{resolution_name}.csv"
        output_df = pd.read_csv(output_path, dtype=str, keep_default_na=False)
        passed = verify_traceability(df, output_df, resolution_name)
        if not passed:
            all_passed = False

    # --- Summary ---
    print(f"\n{'=' * 60}")
    if all_passed:
        print("PHASE 2 COMPLETE | All traceability checks PASSED")
    else:
        print("PHASE 2 COMPLETE | WARNING: Traceability issues detected")
    print(f"{'=' * 60}")

    print(f"\nOutput files:")
    for resolution_name in RESOLUTIONS:
        output_path = OUTPUT_DIR / f"discovery_{resolution_name}.csv"
        out_df = pd.read_csv(output_path, dtype=str, keep_default_na=False)
        n_topics = out_df[out_df["topic_id"] != "-1"]["topic_id"].nunique()
        n_outliers = (out_df["topic_id"] == "-1").sum()
        print(f"  {output_path}: {len(out_df):,} rows, {n_topics} topics, {n_outliers:,} outliers")

    print(f"\nEmbeddings cached: {EMBEDDINGS_CACHE} ({EMBEDDINGS_CACHE.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"\nNext step: Phase 3 - Review discovered topics and build taxonomy")


if __name__ == "__main__":
    main()
