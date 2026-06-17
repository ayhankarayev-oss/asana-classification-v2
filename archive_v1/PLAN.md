# Client Complaint Classification — BERTopic Pipeline Plan

## Overview

Classify 10,000+ Asana client complaints (stored in `name` + `notes` columns of `Client_support.csv`) into a coherent, data-driven issue taxonomy using BERTopic. The goal is to replace or augment the existing hand-crafted tags with categories that actually reflect the complaint distribution.

---

## Phase 0 — Data Audit

**Goal:** Understand the raw data before touching any model.

**Tasks:**
- [ ] Load `Client_support.csv` and inspect shape, columns, nulls
- [ ] Identify which columns carry complaint text (`name`, `notes`, any others)
- [ ] Check existing tags/labels columns — what categories exist today and how many complaints each has
- [ ] Measure text length distribution (short titles vs. long notes)
- [ ] Flag rows where both `name` and `notes` are null/empty
- [ ] Sample 20–30 random rows to understand language style, jargon, language mix

**Output:** Data quality report (printed summary table)

---

## Phase 1 — Environment Setup

**Goal:** Install all dependencies in a clean environment.

```bash
pip install bertopic
pip install sentence-transformers
pip install umap-learn
pip install pandas openpyxl
pip install plotly           # for interactive visualizations
pip install "scikit-learn>=1.3"
pip install openai           # optional: for LLM-enhanced topic labels
```

> **Windows / Python 3.11 note — do NOT `pip install hdbscan`**
> The standalone `hdbscan` package requires C++ build tools and a Cython
> compilation step that routinely fails on Windows with Python 3.11+.
> Instead, use `sklearn.cluster.HDBSCAN` which ships pre-compiled inside
> scikit-learn ≥ 1.3 and is a drop-in replacement for BERTopic's purposes.

**Python version:** 3.9+ recommended (user has Python 3.11)

---

## Phase 2 — Text Preprocessing

**Goal:** Build a single clean `complaint_text` field from `name` + `notes`.

**Steps:**
1. Concatenate: `complaint_text = name + ". " + notes` (handle nulls gracefully)
2. Strip HTML tags if any (Asana notes can contain markdown/HTML)
3. Remove boilerplate phrases (e.g., "Hi team", "Please see below", email signatures)
4. Normalize whitespace and line breaks
5. **Do NOT remove stopwords or stem** — BERTopic's sentence-transformers handle semantics, aggressive preprocessing hurts embedding quality
6. Detect and flag non-English rows (handle separately or translate)

**Output:** Clean `docs` list of strings, one per complaint

---

## Phase 3 — BERTopic Modeling

**Goal:** Discover natural complaint clusters without pre-defining categories.

### 3a — Baseline model (auto topic count)

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import HDBSCAN  # ✓ use sklearn, NOT the hdbscan package

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # fast, good quality

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

# sklearn.cluster.HDBSCAN — no C++ compilation needed, works on Windows/Python 3.11
# Note: sklearn's HDBSCAN does not have prediction_data param — use approximate_predict instead
hdbscan_model = HDBSCAN(
    min_cluster_size=15,     # minimum 15 complaints to form a topic
    metric="euclidean",
    cluster_selection_method="eom",
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    top_n_words=10,
    verbose=True
)

topics, probs = topic_model.fit_transform(docs)
```

### 3b — Review initial topics

```python
# How many topics were found and how big each is
topic_model.get_topic_info()

# Top keywords per topic
for topic_id in topic_model.get_topics():
    print(topic_id, topic_model.get_topic(topic_id))
```

### 3c — Reduce topic count if too granular

```python
# Merge similar topics until ~15–25 remain
topic_model.reduce_topics(docs, nr_topics=20)
topic_model.get_topic_info()
```

**Tuning levers:**
| Parameter | Effect |
|---|---|
| `min_cluster_size` | Larger → fewer, broader topics |
| `nr_topics` | Hard cap on topic count after merging |
| `n_neighbors` (UMAP) | Larger → more global structure |
| `min_topic_size` | Filters out very small clusters |

---

## Phase 4 — Topic Labeling

**Goal:** Turn keyword lists into human-readable issue type names.

### Option A — Manual inspection (fastest)
Review top-10 keywords + 5 sample complaints per topic → assign label yourself.

### Option B — KeyBERT-inspired (automatic, no API needed)

```python
from bertopic.representation import KeyBERTInspired
representation_model = KeyBERTInspired()
topic_model = BERTopic(representation_model=representation_model, ...)
```

### Option C — LLM labels (best quality, requires API key)

```python
from bertopic.representation import OpenAI
import openai

representation_model = OpenAI(
    client=openai.OpenAI(api_key="..."),
    model="gpt-4o-mini",
    chat=True,
    prompt="""I have a topic described by these keywords: [KEYWORDS]
    Sample documents: [DOCUMENTS]
    Give a concise 3-5 word issue type label for this topic. Only return the label."""
)
```

**Suggested label format:** `[Domain] — [Specific Issue]`
Examples:
- `Billing — Incorrect Charge`
- `Access — Account Locked`
- `Integration — Sync Failure`
- `Onboarding — Setup Incomplete`

---

## Phase 5 — Taxonomy Definition

**Goal:** Define the final issue type taxonomy from discovered topics.

**Expected output structure:**

| Topic ID | Label | Keywords | Count | % of Total |
|---|---|---|---|---|
| 0 | Billing — Invoice Error | invoice, charge, wrong, amount... | 842 | 8.4% |
| 1 | Access — Login Failure | login, password, locked, access... | 731 | 7.3% |
| ... | ... | ... | ... | ... |
| -1 | Uncategorized | (outliers) | 412 | 4.1% |

**Hierarchy recommendation:**
- Group topics into 5–8 **top-level domains** (e.g., Billing, Access, Product, Support)
- Keep 15–25 **sub-level issue types** for operational routing
- Topic -1 (outliers) → review manually or create "Other" bucket

---

## Phase 6 — Assign Labels to All Complaints

**Goal:** Write the predicted issue type back to each complaint row.

```python
# Map topic IDs to human labels
topic_labels = {
    0: "Billing — Invoice Error",
    1: "Access — Login Failure",
    # ... fill from Phase 4
    -1: "Uncategorized"
}

df["predicted_topic_id"] = topics
df["predicted_issue_type"] = df["predicted_topic_id"].map(topic_labels)
df["topic_confidence"] = probs.max(axis=1)  # confidence score 0–1

df.to_csv("complaints_classified.csv", index=False)
```

---

## Phase 7 — Visualization & Reporting

**Goal:** Make findings presentable for stakeholders.

```python
# Interactive 2D topic map
topic_model.visualize_topics().write_html("topic_map.html")

# Topic hierarchy (for merging decisions)
topic_model.visualize_hierarchy().write_html("topic_hierarchy.html")

# Topic size bar chart
topic_model.visualize_barchart(top_n_topics=20).write_html("topic_barchart.html")

# Per-complaint topic distribution heatmap
topic_model.visualize_heatmap().write_html("topic_heatmap.html")
```

**Summary report should include:**
- Total complaints analyzed
- Number of topics discovered
- Top 10 issue types by volume
- % of complaints that are "Uncategorized" (Topic -1)
- Comparison: existing Asana tags vs. BERTopic-discovered topics

---

## Phase 8 — Validation

**Goal:** Sanity-check that topics make sense.

- [ ] Sample 10 complaints from each topic — do they share the same issue?
- [ ] Check if existing Asana tags correlate with predicted topics (if labels exist)
- [ ] Spot-check Topic -1 outliers — any coherent sub-group hiding there?
- [ ] Adjust `min_cluster_size` or `nr_topics` if topics feel too broad/narrow
- [ ] Get domain expert review of final taxonomy labels

---

## Phase 9 — (Optional) Production Classifier

**Goal:** Score new incoming complaints automatically as they arrive.

Once taxonomy is validated, train a lightweight supervised classifier on the BERTopic-labeled data:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

clf = Pipeline([
    ("tf", TfidfVectorizer(max_features=20000)),
    ("lr", LogisticRegression(max_iter=1000, C=5))
])

# Use BERTopic labels as ground truth
labeled = df[df["predicted_topic_id"] != -1]
clf.fit(labeled["complaint_text"], labeled["predicted_issue_type"])

# Save for new complaint inference
import joblib
joblib.dump(clf, "issue_classifier.pkl")
```

Or use BERTopic's built-in `transform()` for new documents:
```python
new_topics, new_probs = topic_model.transform(new_docs)
```

---

## Deliverables

| File | Description |
|---|---|
| `complaints_classified.csv` | All complaints with predicted issue type + confidence |
| `topic_map.html` | Interactive 2D visualization of all topics |
| `topic_hierarchy.html` | Hierarchical topic tree |
| `topic_barchart.html` | Volume per topic bar chart |
| `taxonomy.csv` | Final issue type taxonomy: ID, label, keywords, count |
| `issue_classifier.pkl` | (Optional) Trained classifier for new complaints |

---

## File Checklist

- [ ] `Client_support.csv` — place in `C:\Users\Lenovo\Desktop\asana classification\`
- [ ] `bertopic_pipeline.py` — main script (to be written in Phase 3)
- [ ] `data_audit.py` — Phase 0 inspection script
- [ ] `requirements.txt` — pinned dependencies

---

## Decision Points (need your input)

1. **Language:** Are complaints in English only, or mixed?
2. **Desired taxonomy depth:** Do you want broad categories (5–8) or granular issue types (15–25)?
3. **LLM labels:** Do you have an OpenAI/Anthropic API key for automatic topic naming? (Or manual labeling is fine)
4. **Existing tags:** Should we keep, replace, or augment the current Asana classification tags?
5. **Output destination:** Should classified results go back into Asana, or a separate CSV/dashboard?

---

## Timeline Estimate

| Phase | Effort | Blocker |
|---|---|---|
| 0 — Data Audit | 30 min | Need `Client_support.csv` |
| 1 — Setup | 15 min | Python + pip access |
| 2 — Preprocessing | 1 hr | Data quality findings |
| 3 — BERTopic Modeling | 1–2 hr | GPU speeds it up significantly |
| 4 — Topic Labeling | 1 hr | Domain knowledge / API key |
| 5 — Taxonomy Definition | 30 min | Stakeholder review |
| 6 — Label Assignment | 30 min | Finalized taxonomy |
| 7 — Visualization | 30 min | |
| 8 — Validation | 1 hr | Domain expert availability |
| 9 — (Optional) Classifier | 1 hr | |
| **Total** | **~8 hrs** | |
