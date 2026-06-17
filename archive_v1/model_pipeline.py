"""
Phase 3 & 4 - BERTopic Clustering + Explainability Export
===========================================================
Reads : corpus_clean.csv
Writes: complaints_classified.csv   (every row with topic assignment + confidence)
        topic_explanations.csv      (every topic with top-10 c-TF-IDF keywords)

Pipeline
--------
Step 1  Embed documents with Sentence-Transformers (all-MiniLM-L6-v2)
Step 2  Reduce dimensionality with UMAP (768-dim -> 5-dim)
Step 3  Cluster with sklearn HDBSCAN (no C++ compilation required)
Step 4  Build topic profiles with c-TF-IDF + MMR keyword extraction
Step 5  Auto-discover initial topics (no pre-set k required)
Step 6  Hierarchically reduce to exactly 10 business-level issue types
Step 7  Export explainability outputs
"""

import datetime
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------
# Config  (edit these before running)
# --------------------------------------------------------------------------
INPUT_FILE        = Path("corpus_clean.csv")
OUTPUT_CLASSIFIED = Path("complaints_classified.csv")
OUTPUT_EXPLAIN    = Path("topic_explanations.csv")
OUTPUT_REPORT     = Path("cluster_summary_report.md")

# No hard cap on topics — let the model find the natural number.
# reduce_topics() will only fire if initial count exceeds this ceiling.
TARGET_TOPICS     = 20
MIN_CLUSTER_SIZE  = 40    # for the unmatched-doc HDBSCAN pass
TOP_N_WORDS       = 10
MMR_DIVERSITY     = 0.3
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
RANDOM_STATE      = 42

# 12 zero-shot seeds derived from Topic 0 inspection + domain audit.
# Key design rule: use action verbs + domain nouns so the embedding
# matches the task text (which is action-oriented), not just category labels.
# Seeds are intentionally verbose — longer descriptions embed more richly.
ZEROSHOT_TOPICS = [
    # ── New account & bank feed setup ──────────────────────────────────────
    # Platform names removed: sentence-transformer matches on operation semantics,
    # not on "Addepar"/"Arch" which are masked to [PORTFOLIO_PLATFORM] in corpus.
    "new bank brokerage custodian account setup connection feed link online portal access",

    # ── Account maintenance & attribute updates ─────────────────────────────
    "account update maintenance close rename attribute change classification portfolio platform",

    # ── New private investment entry ─────────────────────────────────────────
    "new private investment fund create entry position portfolio platform setup structure direct owner",

    # ── Private investment data updates & valuations ─────────────────────────
    "investment update valuation price change commitment amount capital unfunded transaction",

    # ── Capital calls & individual distribution processing ───────────────────
    "capital call distribution custodian payment processing transfer wire verify confirm receipt",

    # ── Weekly recurring capital call audit (captures 'Check all Arch capital calls') ─
    # Dedicated seed: 523 recurring instances of this exact task were fragmented
    # across 7 topics when using the generic capital calls seed above.
    "weekly periodic check all pending capital calls verify status mark complete audit portfolio platform",

    # ── Data quality & cost basis fixes ─────────────────────────────────────
    "missing data attribute cost basis quality incorrect error fix export reconcile",

    # ── Statement upload & client billing ────────────────────────────────────
    "statement document upload file retrieve download archive K1 invoice send billing document platform",

    # ── Reporting & dashboard views ──────────────────────────────────────────
    "report PDF reporting portfolio platform view dashboard performance deliver generate quarterly",

    # ── Ownership structure & entity changes ─────────────────────────────────
    "ownership structure entity LLC trust beneficiary direct owner setup change legal",

    # ── Transaction reconciliation & recurring platform audits ───────────────
    "recurring weekly monthly transaction reconciliation audit verify confirm status accounting platform operational",

    # ── Onboarding new household ─────────────────────────────────────────────
    "onboarding new household banker introduction portal access setup client portal",

    # ── Follow-up, escalations & coordination ────────────────────────────────
    "follow up reach out email response pending waiting reminder investigate confirm escalation",

    # ── Portfolio view & access configuration ────────────────────────────────
    # View/viewset creation, portal user setup, permission grants — distinct from
    # "reporting" (finished reports) and "account setup" (data feed connections).
    "portfolio platform view viewset create configure user access permissions data aggregator document",

    # ── General operational ad-hoc tasks ─────────────────────────────────────
    "general update ad hoc miscellaneous operational priority task specify",
]

ZEROSHOT_MIN_SIMILARITY = 0.30

# Human-readable business labels mapped to each discovered topic ID.
# Edit these labels here to update both complaints_classified.csv and the report.
BUSINESS_LABELS = {
    # Topic IDs come from the latest BERTopic run (15 topics, 0-14).
    # Labels were assigned by reading cluster_summary_report.md: top c-TF-IDF keywords
    # and the three representative tasks closest to each cluster centroid.
    -1: "Outliers / Unclassified",
     0: "New Account & Data Feed Setup",          # accounts, data feed, connection, online account
     1: "Portfolio Platform Account Updates",      # accounts, support requests, dfo support; rep: "Addepar updates"
     2: "New Private Investment Entry",            # private investment, direct owner, commitment
     3: "Private Investment Updates & Valuations", # series, unfunded, valuations, equity
     4: "Trust & Estate Cash Flow Verification",   # billing, non-exempt marital, transfers, cash flows; rep: Schmulen cash flows / Noble Mortgage
     5: "Capital Call Audit & Monthly Statement Review", # checking, capital, audit, brokerage; rep: TOS asset class review / PJS monthly statements
     6: "Cost Basis & Data Quality Fixes",         # cost basis (0.1995), export, reporting; rep: Download Goldman cost basis
     7: "Document Upload & Client Billing",        # upload, document, prepare, llog; rep: Prepare and send client invoices
     8: "Reporting & Performance Analytics",       # reports, performance, benchmark, quarterly; rep: St. andrews report / HMI reports
     9: "Ownership Structure & Legal Entity Changes", # ownership (0.1093), legal entity (0.1044); rep: New legal entity / Dissolved Oklahoma Entities
    10: "Real Asset Transaction Audit",            # audit, transaction, oil, cash flows, real assets; rep: Armanino Real Asset cash flows
    11: "Loan & Lending Account Setup",            # lender, loan payment, loans; rep: Ricardo Nazario New loan
    12: "Direct Deal Updates & Unlinked Accounts", # sale, direct deals, unlinked, statements; rep: update unlinked direct deals
    13: "Portfolio Platform View & Access Configuration", # access (0.1136), views, columns, filter; rep: Addepar views for Bellco / View Access
    14: "General & Ad Hoc Requests",              # flow, inquiries, general updates; rep: Robert McGill General updates
}

# --------------------------------------------------------------------------
# Load corpus
# --------------------------------------------------------------------------
if not INPUT_FILE.exists():
    sys.exit("ERROR: '{}' not found. Run preprocess.py first.".format(INPUT_FILE))

print("Loading {} ...".format(INPUT_FILE))
df   = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False)
docs = df["complaint_text"].tolist()
print("  {:,} documents loaded".format(len(docs)))

# --------------------------------------------------------------------------
# Step 1-4 - Build BERTopic model components
# --------------------------------------------------------------------------
print("\n[STEP 1-4] Building BERTopic model components ...")

# -- Sentence-Transformer embedding model --
# all-MiniLM-L6-v2: 384-dim embeddings, fast inference, strong semantic quality
# on ~4k documents this takes ~30-60 seconds on CPU
print("  Loading sentence-transformer: {} ...".format(EMBEDDING_MODEL))
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# -- UMAP dimensionality reduction --
# Reduces 384-dim embeddings to 5 dense dimensions before clustering.
# n_components=5 is the BERTopic standard - low enough for HDBSCAN to work
# well, high enough to preserve local neighbourhood structure.
# min_dist=0.0 keeps points tight inside clusters (better HDBSCAN separation).
from umap import UMAP
umap_model = UMAP(
    n_neighbors  = 20,      # 15→20: smoother manifold, reduces micro-fragmentation of identical tasks
    n_components = 5,       # output dimensions fed into HDBSCAN
    min_dist     = 0.0,     # tight clusters: better for HDBSCAN density estimates
    metric       = "cosine",
    random_state = RANDOM_STATE,
    low_memory   = False,
)

# -- HDBSCAN clustering (sklearn build - no C++ compilation on Windows) --
# Uses sklearn.cluster.HDBSCAN (available in scikit-learn >= 1.3).
# min_cluster_size=15 means a topic needs at least 15 complaints.
# cluster_selection_method="eom" (Excess of Mass) is more robust than "leaf"
# and produces more stable, appropriately-sized clusters.
from sklearn.cluster import HDBSCAN
hdbscan_model = HDBSCAN(
    min_cluster_size        = MIN_CLUSTER_SIZE,
    metric                  = "euclidean",
    cluster_selection_method= "eom",
)

# -- CountVectorizer for c-TF-IDF keyword extraction --
# ngram_range=(1,2) captures both single words AND bigrams like
# "missing data", "private investment", "account update" as single features.
# min_df=5: a term must appear in at least 5 documents to be generated.
# max_df=0.90: terms present in >90% of all docs are too generic to be useful.
#
# ANONYMIZATION STOPWORDS: preprocess_v2.py replaces identifiers with typed
# placeholders. These are informative for the sentence-transformer embedding
# step but must be excluded from c-TF-IDF keyword extraction, otherwise the
# keyword list is dominated by placeholder tokens instead of operational terms.
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
_ANON_STOPWORDS = [
    # Typed platform placeholders written by preprocess_v2.py
    "portfolio_platform", "document_platform", "data_aggregator",
    "client_portal", "crm_platform", "accounting_platform", "pm_platform",
    "custodian", "bank", "advisor_org",
    # Entity placeholders
    "client", "employee", "email", "entity_id", "phone", "url",
    # Sub-tokens that appear when brackets are stripped by tokenizer
    "platform", "portal",
    # Domain action verbs: appear in >500 tasks across ALL topic types —
    # zero discriminative signal for c-TF-IDF; removing sharpens keywords.
    "check", "review", "update", "confirm", "please", "task", "status",
    "ensure", "make", "need", "complete", "add", "request", "send",
    "reach", "follow", "provide", "look", "create", "get", "run",
    "using", "use", "based", "per", "per", "see", "note", "below",
    # Form boilerplate tokens
    "general", "updates", "specify", "select", "priority", "level",
    "submitted", "form", "assist", "address", "name",
]
_STOP = list(set(list(ENGLISH_STOP_WORDS) + _ANON_STOPWORDS))
vectorizer_model = CountVectorizer(
    stop_words    = _STOP,
    ngram_range   = (1, 2),
    min_df        = 5,
    max_df        = 0.90,
    token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b",
)

# -- MMR (Maximal Marginal Relevance) representation model --
# After c-TF-IDF finds the top-N candidate keywords, MMR re-ranks them to
# balance RELEVANCE (how defining the word is for the topic) vs DIVERSITY
# (how different consecutive keywords are from each other).
# diversity=0.3: 30% push toward diversity - keeps keywords informative but
# avoids lists like ["account","accounts","accounting","accountant","acct"].
from bertopic.representation import MaximalMarginalRelevance
representation_model = MaximalMarginalRelevance(diversity=MMR_DIVERSITY)

# -- Assemble BERTopic (Zero-Shot mode) --
# With zeroshot_topic_list, BERTopic works in two passes:
#   Pass 1: match every document to the nearest seed topic by cosine similarity.
#           Documents above ZEROSHOT_MIN_SIMILARITY get assigned to a seed topic.
#   Pass 2: remaining unmatched documents are clustered by UMAP+HDBSCAN.
#           These produce extra auto-discovered topics (if any dense clusters exist).
# Using a low ZEROSHOT_MIN_SIMILARITY (0.30) keeps the outlier rate manageable
# for this narrow-domain corpus.
from bertopic import BERTopic
topic_model = BERTopic(
    embedding_model      = embedding_model,
    umap_model           = umap_model,
    hdbscan_model        = hdbscan_model,
    vectorizer_model     = vectorizer_model,
    representation_model = representation_model,
    zeroshot_topic_list  = ZEROSHOT_TOPICS,
    zeroshot_min_similarity = ZEROSHOT_MIN_SIMILARITY,
    top_n_words          = TOP_N_WORDS,
    verbose              = True,
)

# --------------------------------------------------------------------------
# Step 5 - Fit and auto-discover initial topics
# --------------------------------------------------------------------------
print("\n[STEP 5] Fitting BERTopic on {:,} documents ...".format(len(docs)))
print("  Step 5a: Pre-computing embeddings (cached so visualizations skip re-embedding) ...")

embeddings = embedding_model.encode(docs, show_progress_bar=True)
print("  Embeddings cached: {:,} docs x {:,} dims".format(*embeddings.shape))

print("  Step 5b: Fitting ZeroShot BERTopic (UMAP + HDBSCAN + c-TF-IDF) ...")
topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

topic_info_initial = topic_model.get_topic_info()
n_initial = len(topic_info_initial[topic_info_initial["Topic"] != -1])
n_outliers_initial = int((np.array(topics) == -1).sum())

print("\n  Initial discovery results:")
print("  Topics found     : {:,}".format(n_initial))
print("  Outlier docs (-1): {:,} ({:.1f}%)".format(
    n_outliers_initial, n_outliers_initial / len(docs) * 100))
print("\n  Initial topic overview:")
print("  {:<10} {:<12} {}".format("Topic ID", "Size", "Top 5 keywords"))
print("  " + "-" * 60)
for _, row in topic_info_initial.head(20).iterrows():
    tid    = int(row["Topic"])
    tsize  = int(row["Count"])
    label  = str(row.get("Name", ""))[:55]
    print("  {:<10} {:<12,} {}".format(tid, tsize, label))
if len(topic_info_initial) > 20:
    print("  ... ({} more topics not shown)".format(len(topic_info_initial) - 20))

# --------------------------------------------------------------------------
# Step 6 - Hierarchical reduction to TARGET_TOPICS
# --------------------------------------------------------------------------
if n_initial <= TARGET_TOPICS:
    print("\n[STEP 6] {} topics found - within target. No reduction needed.".format(n_initial))
else:
    print("\n[STEP 6] {} topics found (includes auto-discovered extras above the 10 seeds).".format(n_initial))
    print("         Merging down to {} via hierarchical merging ...".format(TARGET_TOPICS))
    # Hierarchical merging absorbs any auto-discovered clusters from the
    # unmatched-document pass back into the nearest zero-shot seed topic.
    topic_model.reduce_topics(docs, nr_topics=TARGET_TOPICS)
    print("  Reduction complete.")

# Retrieve final topic assignments (updated in-place by reduce_topics)
final_topics = topic_model.topics_
topic_info   = topic_model.get_topic_info()

n_final    = len(topic_info[topic_info["Topic"] != -1])
n_outliers = int((np.array(final_topics) == -1).sum())

print("\n  Final topic overview ({} topics + {} outliers):".format(n_final, n_outliers))
print("  {:<10} {:<12} {}".format("Topic ID", "Size", "Top 5 keywords"))
print("  " + "-" * 60)
for _, row in topic_info.iterrows():
    tid   = int(row["Topic"])
    tsize = int(row["Count"])
    label = str(row.get("Name", ""))[:55]
    print("  {:<10} {:<12,} {}".format(tid, tsize, label))

# --------------------------------------------------------------------------
# Build lookup tables for explainability columns
# --------------------------------------------------------------------------
print("\n[STEP 7] Building explainability tables ...")

# For each topic, get the top-N (word, score) pairs from the c-TF-IDF model
topic_ids = [int(r["Topic"]) for _, r in topic_info.iterrows()]

# topic_keywords_map : {topic_id -> [word, word, ...]}  (top N words only)
# topic_scores_map   : {topic_id -> [score, score, ...]}
topic_keywords_map = {}
topic_scores_map   = {}

for tid in topic_ids:
    if tid == -1:
        topic_keywords_map[-1] = ["outlier"] * TOP_N_WORDS
        topic_scores_map[-1]   = [0.0]       * TOP_N_WORDS
        continue
    kw_score_pairs = topic_model.get_topic(tid)    # [(word, score), ...]
    words  = [w  for w, _ in kw_score_pairs]
    scores = [s  for _, s in kw_score_pairs]
    # Pad to TOP_N_WORDS if a topic has fewer keywords
    while len(words)  < TOP_N_WORDS:
        words.append("")
        scores.append(0.0)
    topic_keywords_map[tid] = words[:TOP_N_WORDS]
    topic_scores_map[tid]   = scores[:TOP_N_WORDS]

# topic_label_map : {topic_id -> "kw1 | kw2 | kw3"}
topic_label_map = {
    tid: " | ".join(w for w in topic_keywords_map[tid][:3] if w)
    for tid in topic_ids
}

# topic_top_kws_map : {topic_id -> "kw1, kw2, ..., kw10"}
topic_top_kws_map = {
    tid: ", ".join(w for w in topic_keywords_map[tid] if w)
    for tid in topic_ids
}

# --------------------------------------------------------------------------
# Confidence scores
# --------------------------------------------------------------------------
# HDBSCAN assigns each point a probability (0.0-1.0) representing how
# strongly it belongs to its assigned cluster.
# Noise points (topic_id == -1) always receive a probability of 0.0.
# These values are stored on the fitted hdbscan_model after fit_transform.

if hasattr(topic_model.hdbscan_model, "probabilities_") and \
   len(topic_model.hdbscan_model.probabilities_) == len(docs):
    confidence_scores = list(topic_model.hdbscan_model.probabilities_)
else:
    # Fallback: 1.0 for assigned docs, 0.0 for outliers
    print("  NOTE: HDBSCAN probabilities not available, using binary confidence.")
    confidence_scores = [0.0 if t == -1 else 1.0 for t in final_topics]

# Round to 4 decimal places for readability
confidence_scores = [round(float(c), 4) for c in confidence_scores]

# --------------------------------------------------------------------------
# Build complaints_classified.csv
# --------------------------------------------------------------------------
print("  Building complaints_classified.csv ...")

df_out = df.copy()
df_out["topic_id"]         = final_topics
df_out["is_outlier"]       = [t == -1 for t in final_topics]
df_out["topic_label"]      = [topic_label_map.get(t, "unknown") for t in final_topics]
df_out["top_keywords"]     = [topic_top_kws_map.get(t, "") for t in final_topics]
df_out["confidence_score"] = confidence_scores
df_out["business_label"]   = [BUSINESS_LABELS.get(t, "Unknown") for t in final_topics]

df_out.to_csv(OUTPUT_CLASSIFIED, index=False, encoding="utf-8-sig")
print("  Saved -> {}  ({:,} rows)".format(OUTPUT_CLASSIFIED, len(df_out)))

# --------------------------------------------------------------------------
# Build topic_explanations.csv
# --------------------------------------------------------------------------
# One row per topic. Columns:
#   topic_id, topic_label, topic_size, pct_of_total, pct_of_classified,
#   kw_01..kw_10, ctfidf_01..ctfidf_10,
#   sample_name_1..3   (Name of 3 representative documents)
# --------------------------------------------------------------------------
print("  Building topic_explanations.csv ...")

classified_total = int((np.array(final_topics) != -1).sum())
rows_explain = []

for tid in sorted(topic_ids):
    size = int(topic_info[topic_info["Topic"] == tid]["Count"].values[0])
    pct_total      = round(size / len(docs)       * 100, 2)
    pct_classified = round(size / max(classified_total, 1) * 100, 2) if tid != -1 else 0.0

    kws    = topic_keywords_map[tid]
    scores = topic_scores_map[tid]

    # Pull 3 representative document Names from this topic
    indices = [i for i, t in enumerate(final_topics) if t == tid][:3]
    samples = [df.iloc[i]["Name"][:80] if i < len(df) else "" for i in indices]
    while len(samples) < 3:
        samples.append("")

    row = {
        "topic_id"         : tid,
        "topic_label"      : topic_label_map[tid],
        "topic_size"       : size,
        "pct_of_total"     : pct_total,
        "pct_of_classified": pct_classified,
    }
    # Keyword columns with zero-padded rank: kw_01, kw_02 ... kw_10
    for rank in range(TOP_N_WORDS):
        row["kw_{:02d}".format(rank + 1)]       = kws[rank]
        row["ctfidf_{:02d}".format(rank + 1)]   = round(float(scores[rank]), 6)

    row["sample_name_1"] = samples[0]
    row["sample_name_2"] = samples[1]
    row["sample_name_3"] = samples[2]

    rows_explain.append(row)

df_explain = pd.DataFrame(rows_explain)
df_explain.to_csv(OUTPUT_EXPLAIN, index=False, encoding="utf-8-sig")
print("  Saved -> {}  ({} topics)".format(OUTPUT_EXPLAIN, len(df_explain)))

# --------------------------------------------------------------------------
# Step 8 - Generate cluster_summary_report.md
# --------------------------------------------------------------------------
print("\n[STEP 8] Generating cluster_summary_report.md ...")

def _kw_distinctiveness(score):
    if score >= 0.10:
        return "Very high -- strongly unique to this cluster"
    elif score >= 0.05:
        return "High -- clearly characteristic"
    elif score >= 0.02:
        return "Moderate -- shares vocabulary with adjacent topics"
    else:
        return "Low -- diffuse, overlaps many topics"


def _build_summary_paragraph(label, size, pct_total, kws, scores, rep_names):
    top_kw    = kws[0]    if kws    else "unknown"
    top_score = scores[0] if scores else 0.0

    # Scale descriptor
    if pct_total > 15:
        size_desc  = "one of the largest operational areas in the corpus"
        scale_note = (
            "At {:.1f}% of total task volume, this is one of the highest-demand categories "
            "in the portfolio and is a strong candidate for workflow automation, task templates, "
            "or dedicated SLA tracking to manage recurring throughput at scale.".format(pct_total)
        )
    elif pct_total > 7:
        size_desc  = "a major operational area"
        scale_note = (
            "At {:.1f}% of total volume, this represents a significant and recurring stream "
            "of work that warrants standardised procedures, clear ownership, and defined "
            "turnaround expectations.".format(pct_total)
        )
    elif pct_total > 3:
        size_desc  = "a regular operational category"
        scale_note = (
            "At {:.1f}% of total volume, this is a steady category where consistent "
            "processes and clear handoff points will improve throughput.".format(pct_total)
        )
    else:
        size_desc  = "a focused specialist category"
        scale_note = (
            "Although smaller in volume ({:.1f}%), this cluster represents specialist or "
            "high-touch work that may require dedicated expertise, third-party coordination, "
            "or specific regulatory awareness.".format(pct_total)
        )

    # Keyword quality commentary
    if top_score >= 0.10:
        kw_note = (
            "The dominant keyword **{}** carries an exceptionally high c-TF-IDF score of {:.4f}, "
            "confirming this is a tightly cohesive, well-defined operational category "
            "with strongly consistent vocabulary across every grouped task.".format(top_kw, top_score)
        )
    elif top_score >= 0.05:
        kw_note = (
            "The leading keyword **{}** (c-TF-IDF: {:.4f}) is clearly characteristic of this "
            "cluster, reflecting consistent vocabulary and a recognisable operational theme "
            "across the tasks grouped here.".format(top_kw, top_score)
        )
    elif top_score >= 0.02:
        kw_note = (
            "The top keyword **{}** (c-TF-IDF: {:.4f}) provides moderate differentiation. "
            "The cluster has a recognisable theme but shares vocabulary with neighbouring "
            "categories -- expected behaviour in a narrow, domain-specific operations corpus "
            "where most tasks reference the same core systems (Addepar, Arch, Egnyte).".format(top_kw, top_score)
        )
    else:
        kw_note = (
            "The low c-TF-IDF scores across all keywords (top: **{}** at {:.4f}) indicate "
            "that this cluster's vocabulary is diffuse and overlaps heavily with other categories. "
            "This is a genuine catch-all bucket: tasks here share no single dominant theme "
            "and resisted more granular clustering despite multiple seeding iterations. "
            "Manual review by a domain expert is recommended to surface any hidden "
            "sub-categories.".format(top_kw, top_score)
        )

    # Representative task sentence
    clean_names = [n[:90].strip() for n in rep_names[:2] if n.strip()]
    if len(clean_names) == 2:
        rep_sent = (
            "Representative tasks include \"{}\" and \"{}\".".format(
                clean_names[0], clean_names[1])
        )
    elif len(clean_names) == 1:
        rep_sent = "A representative task is \"{}\".".format(clean_names[0])
    else:
        rep_sent = ""

    return " ".join(p for p in [
        "This cluster is {} covering {:,} tasks ({:.1f}% of the full corpus).".format(
            size_desc, size, pct_total),
        kw_note,
        rep_sent,
        scale_note,
    ] if p)


# Retrieve representative documents from BERTopic.
# get_representative_docs() returns the complaint_text strings that lie
# geometrically closest to each topic centroid in the reduced embedding space.
# We map each back to its original task Name for a more readable report.
try:
    rep_docs_raw = topic_model.get_representative_docs()
except Exception:
    rep_docs_raw = {}

text_to_name = {row["complaint_text"]: row["Name"] for _, row in df.iterrows()}

def _rep_names_for(tid):
    rep_texts = rep_docs_raw.get(tid, [])
    names = []
    seen  = set()
    for text in rep_texts:
        name = text_to_name.get(text, text[:100]).strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    # Supplement with sequential docs from the topic until we have 3 unique examples
    if len(names) < 3:
        indices = [i for i, t in enumerate(final_topics) if t == tid]
        for i in indices:
            if len(names) >= 3:
                break
            if i < len(df):
                name = df.iloc[i]["Name"][:80].strip()
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
    return names


# Sort topics by size descending; outliers always last
topic_order_report = sorted(
    [t for t in topic_ids if t != -1],
    key=lambda t: int(topic_info[topic_info["Topic"] == t]["Count"].values[0]),
    reverse=True,
)
topic_order_report.append(-1)

today_str      = datetime.date.today().strftime("%Y-%m-%d")
total_docs_n   = len(docs)
pct_outliers_r = n_outliers / total_docs_n * 100

# Executive summary text
top3 = topic_order_report[:3]
top3_labels = [BUSINESS_LABELS.get(t, str(t)) for t in top3]
top3_pcts   = [
    round(int(topic_info[topic_info["Topic"] == t]["Count"].values[0])
          / total_docs_n * 100, 1)
    for t in top3
]
top3_combined = sum(top3_pcts)

exec_summary_text = (
    "The BERTopic model identified {n} operationally distinct clusters from a corpus of "
    "{total:,} Asana tasks. The three highest-volume categories -- **{t1}** ({p1:.1f}%), "
    "**{t2}** ({p2:.1f}%), and **{t3}** ({p3:.1f}%) -- together account for {combined:.0f}% "
    "of all task volume. The model assigned {pct_cls:.1f}% of tasks to a named cluster, "
    "leaving only {pct_out:.1f}% as outliers requiring manual review. "
    "Cluster profiles below are sorted by volume (largest first) so the highest-priority "
    "operational areas appear at the top.".format(
        n=n_final, total=total_docs_n,
        t1=top3_labels[0], p1=top3_pcts[0],
        t2=top3_labels[1], p2=top3_pcts[1],
        t3=top3_labels[2], p3=top3_pcts[2],
        combined=top3_combined,
        pct_cls=round(100 - pct_outliers_r, 1),
        pct_out=round(pct_outliers_r, 1),
    )
)

md = []
md.append("# TOS Asana Task Cluster Summary Report")
md.append("")
md.append("| | |")
md.append("|---|---|")
md.append("| **Report date** | {} |".format(today_str))
md.append("| **Pipeline** | BERTopic v0.17 - ZeroShot mode - all-MiniLM-L6-v2 embeddings |")
md.append("| **Corpus** | {:,} tasks from Client_support.csv |".format(total_docs_n))
md.append("| **Topics found** | {} clusters + outlier bin |".format(n_final))
md.append("| **Coverage** | {:.1f}% classified / {:.1f}% outliers |".format(
    round(100 - pct_outliers_r, 1), round(pct_outliers_r, 1)))
md.append("")
md.append("---")
md.append("")
md.append("## Executive Summary")
md.append("")
md.append(exec_summary_text)
md.append("")
md.append("---")
md.append("")
md.append("## Cluster Profiles")
md.append("")
md.append(
    "Each profile shows: (1) the mathematical keyword basis for the cluster via c-TF-IDF, "
    "(2) the three tasks geometrically closest to the cluster centroid in embedding space, "
    "and (3) an operational interpretation of what the data tells us about real work "
    "happening inside that bucket."
)
md.append("")

for tid in topic_order_report:
    lbl      = BUSINESS_LABELS.get(tid, "Unknown")
    size_row = topic_info[topic_info["Topic"] == tid]
    if size_row.empty:
        continue
    t_size   = int(size_row["Count"].values[0])
    pct_tot  = round(t_size / total_docs_n * 100, 1)
    pct_cls  = round(t_size / max(classified_total, 1) * 100, 1) if tid != -1 else 0.0

    kws    = topic_keywords_map[tid]
    scores = topic_scores_map[tid]

    md.append("---")
    md.append("")
    if tid == -1:
        md.append("## Outlier Bin -- {}".format(lbl))
    else:
        md.append("## Topic {} -- {}".format(tid, lbl))
    md.append("")
    md.append(
        "*Cluster ID: {} | {:,} tasks | {:.1f}% of total corpus | "
        "{:.1f}% of classified tasks*".format(tid, t_size, pct_tot, pct_cls)
    )
    md.append("")

    if tid != -1:
        # ── Keywords table ──────────────────────────────────────────────
        md.append("### Defining Keywords (c-TF-IDF)")
        md.append("")
        md.append("| Rank | Keyword | Score | Distinctiveness |")
        md.append("|:----:|---------|------:|-----------------|")
        for rank, (kw, sc) in enumerate(zip(kws, scores), 1):
            if not kw:
                continue
            md.append("| {} | {} | {:.4f} | {} |".format(
                rank, kw, sc, _kw_distinctiveness(sc)))
        md.append("")

        # ── Representative tasks ─────────────────────────────────────────
        rep_names = _rep_names_for(tid)
        md.append("### Representative Tasks")
        md.append("")
        md.append(
            "The three tasks geometrically closest to this cluster's centroid "
            "in sentence-embedding space (i.e., most archetypal examples of this category):"
        )
        md.append("")
        for i, name in enumerate(rep_names[:3], 1):
            md.append("{}. {}".format(i, name[:130].strip()))
        md.append("")

        # ── Operational summary ──────────────────────────────────────────
        md.append("### Operational Summary")
        md.append("")
        md.append(_build_summary_paragraph(lbl, t_size, pct_tot, kws, scores, rep_names))
        md.append("")

    else:
        # ── Outlier section ──────────────────────────────────────────────
        md.append("### What Are Outliers?")
        md.append("")
        md.append(
            "These {:,} tasks ({:.1f}% of the corpus) could not be confidently assigned "
            "to any cluster. Common causes: task names that are too short or vague to produce "
            "a reliable embedding, one-off tasks with highly unique vocabulary, or borderline "
            "tasks that scored below the {:.2f} cosine-similarity threshold against all "
            "{} zero-shot seeds. "
            "These tasks are preserved in **complaints_classified.csv** with "
            "`topic_id = -1` and should be reviewed manually or assigned to the nearest "
            "business category by a domain expert.".format(
                t_size, pct_tot,
                ZEROSHOT_MIN_SIMILARITY, len(ZEROSHOT_TOPICS))
        )
        md.append("")

md.append("---")
md.append("")
md.append(
    "*Generated by `model_pipeline.py` using BERTopic v0.17 + ZeroShot topic modelling.*  "
)
md.append(
    "*c-TF-IDF (class-based TF-IDF): scores reflect how uniquely a term identifies one cluster*  "
)
md.append(
    "*relative to the full corpus. Higher = more diagnostic.*"
)

report_text = "\n".join(md)
with open(OUTPUT_REPORT, "w", encoding="utf-8") as fh:
    fh.write(report_text)
print("  Saved -> {}".format(OUTPUT_REPORT))

# --------------------------------------------------------------------------
# Step 9 - Interactive Visualizations  (Phase 7)
# --------------------------------------------------------------------------
print("\n[STEP 9] Generating interactive HTML visualizations ...")

# Wire business labels into the model so every chart shows human-readable names
# (excludes -1 because BERTopic's set_topic_labels does not accept the outlier id)
topic_model.set_topic_labels({k: v for k, v in BUSINESS_LABELS.items() if k != -1})

n_viz_topics = len([t for t in topic_ids if t != -1])

# 1 -- 2D Intertopic Distance Map
# Bubble positions = topic centroids projected to 2D via internal UMAP.
# Bubble size = number of docs in the topic.
# Pass pre-computed embeddings so BERTopic skips a second encode() call.
print("  [1/4] topic_map.html  (2D intertopic distance map) ...")
try:
    fig = topic_model.visualize_topics(
        custom_labels = True,
        title         = "<b>TOS Asana Task Clusters -- Intertopic Distance Map</b>",
    )
    fig.write_html("topic_map.html")
    print("        Saved -> topic_map.html")
except Exception as e:
    print("        SKIP: {}".format(e))

# 2 -- Keyword bar charts (one subplot per topic, top 8 c-TF-IDF words each)
print("  [2/4] topic_barchart.html  (keyword bars per topic) ...")
try:
    fig = topic_model.visualize_barchart(
        top_n_topics  = n_viz_topics,
        n_words       = 8,
        custom_labels = True,
        title         = "<b>TOS Asana -- Top 8 c-TF-IDF Keywords per Topic</b>",
    )
    fig.write_html("topic_barchart.html")
    print("        Saved -> topic_barchart.html")
except Exception as e:
    print("        SKIP: {}".format(e))

# 3 -- Hierarchical topic dendrogram
# Shows how similar topics are and which would merge first under reduction.
print("  [3/4] topic_hierarchy.html  (topic dendrogram) ...")
try:
    fig = topic_model.visualize_hierarchy(
        custom_labels = True,
        title         = "<b>TOS Asana -- Topic Hierarchy (similarity dendrogram)</b>",
    )
    fig.write_html("topic_hierarchy.html")
    print("        Saved -> topic_hierarchy.html")
except Exception as e:
    print("        SKIP: {}".format(e))

# 4 -- Topic-to-topic cosine similarity heatmap
# Darker cell = higher vocabulary overlap between those two topics.
print("  [4/4] topic_heatmap.html  (topic similarity matrix) ...")
try:
    fig = topic_model.visualize_heatmap(
        custom_labels = True,
        title         = "<b>TOS Asana -- Topic Similarity Matrix</b>",
    )
    fig.write_html("topic_heatmap.html")
    print("        Saved -> topic_heatmap.html")
except Exception as e:
    print("        SKIP: {}".format(e))

print("  Done. Open the .html files in any browser -- no server required.")

# --------------------------------------------------------------------------
# Final summary
# --------------------------------------------------------------------------
print("\n" + "=" * 65)
print("MODEL PIPELINE COMPLETE")
print("=" * 65)
print("\n  Documents processed : {:,}".format(len(docs)))
print("  Initial topics found: {:,}".format(n_initial))
print("  Final topics        : {:,}".format(n_final))
print("  Outlier docs (topic -1): {:,} ({:.1f}%)".format(
    n_outliers, n_outliers / len(docs) * 100))

print("\n  FINAL TOPIC SUMMARY")
print("  {:<6} {:<8} {:<7} {}".format("ID", "Size", "% Tot", "Label (top 3 keywords)"))
print("  " + "-" * 62)
for tid in sorted(t for t in topic_ids if t != -1):
    info_row = topic_info[topic_info["Topic"] == tid]
    if info_row.empty:
        continue
    size = int(info_row["Count"].values[0])
    pct  = size / len(docs) * 100
    lbl  = topic_label_map[tid]
    print("  {:<6} {:<8,} {:<7.1f}% {}".format(tid, size, pct, lbl))

# Outliers last
out_row = topic_info[topic_info["Topic"] == -1]
if not out_row.empty:
    out_size = int(out_row["Count"].values[0])
    print("  {:<6} {:<8,} {:<7.1f}% {}".format(
        -1, out_size, out_size / len(docs) * 100, "(outliers / uncategorized)"))

print("\n  Outputs:")
print("    {} - row-by-row classifications with confidence".format(OUTPUT_CLASSIFIED))
print("    {} - per-topic keyword profiles (explainability)".format(OUTPUT_EXPLAIN))
print("    {} - markdown cluster summary report".format(OUTPUT_REPORT))
print("    topic_map.html          - 2D intertopic distance map")
print("    topic_barchart.html     - keyword bars per topic")
print("    topic_hierarchy.html    - topic similarity dendrogram")
print("    topic_heatmap.html      - topic-to-topic similarity matrix")
print("=" * 65)