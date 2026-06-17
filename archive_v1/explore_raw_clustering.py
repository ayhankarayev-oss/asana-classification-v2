"""
Raw Text Clustering Exploration
================================
Reads : corpus_clean.csv  (uses complaint_text_raw — unmasked text)
Writes: raw_classified.csv
        raw_topic_explanations.csv
        raw_cluster_report.md
        raw_topic_map.html
        raw_topic_barchart.html
        raw_topic_hierarchy.html
        raw_topic_heatmap.html

Pipeline
--------
Step 1  Embed with sentence-transformers (all-MiniLM-L6-v2)
Step 2  Reduce dimensionality with UMAP (→ 5 dims)
Step 3  Cluster with HDBSCAN — auto-discovers topic count
Step 4  Extract keywords with c-TF-IDF + MMR
Step 5  Export classified CSV, topic explanations, report, HTML charts

No zero-shot seeds. No masking. Pure data-driven discovery on raw text.
Outputs are prefixed raw_ so existing results are never overwritten.
"""

import datetime
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE        = Path("corpus_clean.csv")
OUTPUT_CLASSIFIED = Path("raw_classified.csv")
OUTPUT_EXPLAIN    = Path("raw_topic_explanations.csv")
OUTPUT_REPORT     = Path("raw_cluster_report.md")

EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
MIN_CLUSTER_SIZE = 40
TOP_N_WORDS      = 10
MMR_DIVERSITY    = 0.3
RANDOM_STATE     = 42

# ── Load corpus ───────────────────────────────────────────────────────────────
if not INPUT_FILE.exists():
    sys.exit(f"ERROR: '{INPUT_FILE}' not found. Run preprocess.py first.")

print(f"Loading {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False)

# Use unmasked text when available; fall back gracefully
if "complaint_text_raw" in df.columns:
    TEXT_COL = "complaint_text_raw"
else:
    TEXT_COL = "complaint_text"
    print("  NOTE: 'complaint_text_raw' column not found — using 'complaint_text' instead.")
    print("        Run preprocess_v2.py to generate the unmasked column.")

print(f"  Text column : '{TEXT_COL}'")
docs = df[TEXT_COL].tolist()
print(f"  Documents   : {len(docs):,}")

# ── Step 1 — Sentence-transformer embedding ───────────────────────────────────
print(f"\n[STEP 1] Loading sentence-transformer: {EMBEDDING_MODEL} ...")
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# ── Step 2 — UMAP ─────────────────────────────────────────────────────────────
print("[STEP 2] Configuring UMAP (384-dim → 5-dim) ...")
from umap import UMAP
umap_model = UMAP(
    n_neighbors  = 20,
    n_components = 5,
    min_dist     = 0.0,
    metric       = "cosine",
    random_state = RANDOM_STATE,
    low_memory   = False,
)

# ── Step 3 — HDBSCAN ──────────────────────────────────────────────────────────
print("[STEP 3] Configuring HDBSCAN ...")
from sklearn.cluster import HDBSCAN
hdbscan_model = HDBSCAN(
    min_cluster_size         = MIN_CLUSTER_SIZE,
    metric                   = "euclidean",
    cluster_selection_method = "eom",
)

# ── Step 4 — Vectorizer + MMR ─────────────────────────────────────────────────
print("[STEP 4] Configuring CountVectorizer + MMR ...")
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# Domain-wide verbs that appear in nearly every task — no discriminative value
DOMAIN_STOPWORDS = [
    "check", "review", "update", "confirm", "please", "task", "status",
    "ensure", "make", "need", "complete", "add", "request", "send",
    "reach", "follow", "provide", "look", "create", "get", "run",
    "using", "use", "based", "per", "see", "note", "below",
    "general", "updates", "specify", "select", "priority",
    "submitted", "form", "assist", "address",
]
STOP = list(set(list(ENGLISH_STOP_WORDS) + DOMAIN_STOPWORDS))

vectorizer_model = CountVectorizer(
    stop_words    = STOP,
    ngram_range   = (1, 2),
    min_df        = 5,
    max_df        = 0.90,
    token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b",
)

from bertopic.representation import MaximalMarginalRelevance
representation_model = MaximalMarginalRelevance(diversity=MMR_DIVERSITY)

# ── Assemble BERTopic — no zero-shot list ─────────────────────────────────────
print("[STEP 5] Assembling BERTopic (pure HDBSCAN — no zero-shot seeds) ...")
from bertopic import BERTopic
topic_model = BERTopic(
    embedding_model      = embedding_model,
    umap_model           = umap_model,
    hdbscan_model        = hdbscan_model,
    vectorizer_model     = vectorizer_model,
    representation_model = representation_model,
    top_n_words          = TOP_N_WORDS,
    verbose              = True,
)

# ── Encode + fit ──────────────────────────────────────────────────────────────
print(f"\n[STEP 6] Encoding {len(docs):,} documents ...")
embeddings = embedding_model.encode(docs, show_progress_bar=True)
print(f"  Shape: {embeddings.shape[0]:,} × {embeddings.shape[1]:,}")

print("\n[STEP 7] Fitting BERTopic (UMAP → HDBSCAN → c-TF-IDF) ...")
topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

topic_info = topic_model.get_topic_info()
n_topics   = len(topic_info[topic_info["Topic"] != -1])
n_outliers = int((np.array(topics) == -1).sum())

print(f"\n  Topics auto-discovered : {n_topics}")
print(f"  Outlier docs (-1)      : {n_outliers:,} ({n_outliers / len(docs) * 100:.1f}%)")
print(f"\n  {'ID':<8} {'Size':<10} Top keywords")
print("  " + "-" * 70)
for _, row in topic_info.iterrows():
    tid   = int(row["Topic"])
    tsize = int(row["Count"])
    label = str(row.get("Name", ""))[:55]
    print(f"  {tid:<8} {tsize:<10,} {label}")

# ── Confidence scores ─────────────────────────────────────────────────────────
if (hasattr(topic_model.hdbscan_model, "probabilities_") and
        len(topic_model.hdbscan_model.probabilities_) == len(docs)):
    confidence_scores = [round(float(c), 4) for c in topic_model.hdbscan_model.probabilities_]
else:
    print("  NOTE: HDBSCAN probabilities unavailable — using binary confidence.")
    confidence_scores = [0.0 if t == -1 else 1.0 for t in topics]

# ── Keyword maps ──────────────────────────────────────────────────────────────
topic_ids = [int(r["Topic"]) for _, r in topic_info.iterrows()]

topic_keywords_map = {}
topic_scores_map   = {}
for tid in topic_ids:
    if tid == -1:
        topic_keywords_map[-1] = ["outlier"] * TOP_N_WORDS
        topic_scores_map[-1]   = [0.0]       * TOP_N_WORDS
        continue
    pairs  = topic_model.get_topic(tid)
    words  = [w for w, _ in pairs]
    scores = [s for _, s in pairs]
    while len(words) < TOP_N_WORDS:
        words.append("")
        scores.append(0.0)
    topic_keywords_map[tid] = words[:TOP_N_WORDS]
    topic_scores_map[tid]   = scores[:TOP_N_WORDS]

topic_label_map   = {tid: " | ".join(w for w in topic_keywords_map[tid][:3] if w)
                     for tid in topic_ids}
topic_top_kws_map = {tid: ", ".join(w for w in topic_keywords_map[tid] if w)
                     for tid in topic_ids}

# ── raw_classified.csv ────────────────────────────────────────────────────────
print(f"\n[OUTPUT 1/7] {OUTPUT_CLASSIFIED} ...")
df_out = df.copy()
df_out["topic_id"]         = topics
df_out["is_outlier"]       = [t == -1 for t in topics]
df_out["topic_label"]      = [topic_label_map.get(t, "unknown") for t in topics]
df_out["top_keywords"]     = [topic_top_kws_map.get(t, "") for t in topics]
df_out["confidence_score"] = confidence_scores
df_out.to_csv(OUTPUT_CLASSIFIED, index=False, encoding="utf-8-sig")
print(f"  Saved → {OUTPUT_CLASSIFIED}  ({len(df_out):,} rows)")

# ── raw_topic_explanations.csv ────────────────────────────────────────────────
print(f"[OUTPUT 2/7] {OUTPUT_EXPLAIN} ...")
classified_total = int((np.array(topics) != -1).sum())
text_to_name     = {str(row[TEXT_COL]): str(row.get("Name", "")) for _, row in df.iterrows()}


def _rep_names(tid):
    try:
        rep_texts = topic_model.get_representative_docs().get(tid, [])
    except Exception:
        rep_texts = []
    names = []
    seen  = set()
    for text in rep_texts:
        name = text_to_name.get(text, text[:100]).strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    if len(names) < 3:
        for i, t in enumerate(topics):
            if t != tid or len(names) >= 3:
                continue
            name = str(df.iloc[i].get("Name", ""))[:80].strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    while len(names) < 3:
        names.append("")
    return names


rows_explain = []
for tid in sorted(topic_ids):
    size_row = topic_info[topic_info["Topic"] == tid]
    if size_row.empty:
        continue
    size      = int(size_row["Count"].values[0])
    pct_total = round(size / len(docs) * 100, 2)
    pct_cls   = round(size / max(classified_total, 1) * 100, 2) if tid != -1 else 0.0
    kws       = topic_keywords_map[tid]
    scores    = topic_scores_map[tid]
    samples   = _rep_names(tid)

    row = {
        "topic_id"         : tid,
        "topic_label"      : topic_label_map[tid],
        "topic_size"       : size,
        "pct_of_total"     : pct_total,
        "pct_of_classified": pct_cls,
    }
    for rank in range(TOP_N_WORDS):
        row[f"kw_{rank+1:02d}"]     = kws[rank]
        row[f"ctfidf_{rank+1:02d}"] = round(float(scores[rank]), 6)
    row["sample_name_1"] = samples[0]
    row["sample_name_2"] = samples[1]
    row["sample_name_3"] = samples[2]
    rows_explain.append(row)

pd.DataFrame(rows_explain).to_csv(OUTPUT_EXPLAIN, index=False, encoding="utf-8-sig")
print(f"  Saved → {OUTPUT_EXPLAIN}  ({len(rows_explain)} topics)")

# ── raw_cluster_report.md ─────────────────────────────────────────────────────
print(f"[OUTPUT 3/7] {OUTPUT_REPORT} ...")

def _distinctiveness(score):
    if score >= 0.10: return "Very high"
    if score >= 0.05: return "High"
    if score >= 0.02: return "Moderate"
    return "Low"


today   = datetime.date.today().strftime("%Y-%m-%d")
pct_out = n_outliers / len(docs) * 100
pct_cls = 100 - pct_out

topic_order = sorted(
    [t for t in topic_ids if t != -1],
    key=lambda t: int(topic_info[topic_info["Topic"] == t]["Count"].values[0]),
    reverse=True,
)
topic_order.append(-1)

top3        = topic_order[:3]
top3_sizes  = [int(topic_info[topic_info["Topic"] == t]["Count"].values[0]) for t in top3]
top3_pcts   = [round(s / len(docs) * 100, 1) for s in top3_sizes]
top3_labels = [topic_label_map[t] for t in top3]

md = [
    "# Raw Text Cluster Report",
    "",
    "| | |",
    "|---|---|",
    f"| **Report date**  | {today} |",
    f"| **Pipeline**     | BERTopic — pure HDBSCAN, no zero-shot seeds |",
    f"| **Embeddings**   | {EMBEDDING_MODEL} |",
    f"| **Text source**  | `{TEXT_COL}` (unmasked) |",
    f"| **Corpus**       | {len(docs):,} tasks |",
    f"| **Topics found** | {n_topics} clusters + outlier bin |",
    f"| **Coverage**     | {pct_cls:.1f}% classified / {pct_out:.1f}% outliers |",
    "",
    "---",
    "",
    "## Executive Summary",
    "",
    (
        f"BERTopic (pure HDBSCAN, no zero-shot seeds) auto-discovered **{n_topics} clusters** "
        f"from {len(docs):,} unmasked Asana tasks. "
        f"The three largest clusters — "
        f"**{top3_labels[0]}** ({top3_pcts[0]:.1f}%), "
        f"**{top3_labels[1]}** ({top3_pcts[1]:.1f}%), and "
        f"**{top3_labels[2]}** ({top3_pcts[2]:.1f}%) — "
        f"account for {sum(top3_pcts):.0f}% of total volume. "
        f"{pct_cls:.1f}% of tasks were assigned to a named cluster; "
        f"{pct_out:.1f}% remain as outliers. "
        f"Because no masking was applied, person names, client names, and platform "
        f"names are visible in the keyword profiles — topics dominated by a single "
        f"entity name likely reflect client or platform volume bias rather than a "
        f"distinct operational category."
    ),
    "",
    "---",
    "",
    "## Cluster Profiles",
    "",
    "Clusters are sorted by size (largest first). Keywords are c-TF-IDF scores — "
    "how uniquely a term identifies this cluster relative to the full corpus.",
    "",
]

for tid in topic_order:
    size_row = topic_info[topic_info["Topic"] == tid]
    if size_row.empty:
        continue
    t_size  = int(size_row["Count"].values[0])
    pct_tot = round(t_size / len(docs) * 100, 1)
    pct_c   = round(t_size / max(classified_total, 1) * 100, 1) if tid != -1 else 0.0
    kws     = topic_keywords_map[tid]
    scores  = topic_scores_map[tid]
    label   = topic_label_map[tid]

    md += ["---", ""]
    if tid == -1:
        md.append("## Outlier Bin")
    else:
        md.append(f"## Topic {tid} — {label}")
    md.append("")
    md.append(f"*{t_size:,} tasks | {pct_tot:.1f}% of corpus | {pct_c:.1f}% of classified*")
    md.append("")

    if tid != -1:
        md += ["### Keywords (c-TF-IDF)", ""]
        md.append("| Rank | Keyword | Score | Distinctiveness |")
        md.append("|:----:|---------|------:|-----------------|")
        for rank, (kw, sc) in enumerate(zip(kws, scores), 1):
            if kw:
                md.append(f"| {rank} | {kw} | {sc:.4f} | {_distinctiveness(sc)} |")
        md.append("")

        rep = _rep_names(tid)
        md += ["### Representative Tasks", ""]
        for i, name in enumerate(rep[:3], 1):
            md.append(f"{i}. {name[:130].strip()}")
        md.append("")
    else:
        md += [
            f"These {t_size:,} tasks ({pct_tot:.1f}%) fell below HDBSCAN's density "
            f"threshold (`min_cluster_size={MIN_CLUSTER_SIZE}`). Common causes: very "
            f"short task names, one-off vocabulary, or borderline tasks between two clusters.",
            "",
        ]

md += [
    "---",
    "",
    f"*Generated by `explore_raw_clustering.py` · BERTopic pure-HDBSCAN*",
    f"*Text source: `{TEXT_COL}` — unmasked task text*",
]

with open(OUTPUT_REPORT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(md))
print(f"  Saved → {OUTPUT_REPORT}")

# ── HTML visualizations ───────────────────────────────────────────────────────
print("\n[OUTPUT 4-7/7] HTML visualizations ...")

viz_jobs = [
    (
        "raw_topic_map.html",
        lambda: topic_model.visualize_topics(
            title="<b>Raw Clustering — Intertopic Distance Map</b>"
        ),
    ),
    (
        "raw_topic_barchart.html",
        lambda nt=n_topics: topic_model.visualize_barchart(
            top_n_topics=nt,
            n_words=8,
            title="<b>Raw Clustering — Top 8 Keywords per Topic</b>",
        ),
    ),
    (
        "raw_topic_hierarchy.html",
        lambda: topic_model.visualize_hierarchy(
            title="<b>Raw Clustering — Topic Hierarchy</b>"
        ),
    ),
    (
        "raw_topic_heatmap.html",
        lambda: topic_model.visualize_heatmap(
            title="<b>Raw Clustering — Topic Similarity Matrix</b>"
        ),
    ),
]

for fname, fn in viz_jobs:
    try:
        fig = fn()
        fig.write_html(fname)
        print(f"  Saved → {fname}")
    except Exception as e:
        print(f"  SKIP {fname}: {e}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RAW CLUSTERING COMPLETE")
print("=" * 65)
print(f"  Documents        : {len(docs):,}")
print(f"  Topics found     : {n_topics}")
print(f"  Outliers (topic -1): {n_outliers:,} ({pct_out:.1f}%)")
print(f"  Classified       : {classified_total:,} ({pct_cls:.1f}%)")
print(f"\n  {'ID':<6} {'Size':<8} {'%Tot':<7} Top-3 keywords")
print("  " + "-" * 65)
for tid in sorted(t for t in topic_ids if t != -1):
    row = topic_info[topic_info["Topic"] == tid]
    if row.empty:
        continue
    size = int(row["Count"].values[0])
    pct  = size / len(docs) * 100
    print(f"  {tid:<6} {size:<8,} {pct:<7.1f}% {topic_label_map[tid]}")
out_row = topic_info[topic_info["Topic"] == -1]
if not out_row.empty:
    size = int(out_row["Count"].values[0])
    print(f"  {-1:<6} {size:<8,} {size / len(docs) * 100:<7.1f}% (outliers)")
print(f"\n  Outputs written:")
print(f"    {OUTPUT_CLASSIFIED}")
print(f"    {OUTPUT_EXPLAIN}")
print(f"    {OUTPUT_REPORT}")
print(f"    raw_topic_map.html")
print(f"    raw_topic_barchart.html")
print(f"    raw_topic_hierarchy.html")
print(f"    raw_topic_heatmap.html")
print("=" * 65)
