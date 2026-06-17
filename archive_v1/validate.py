"""
Phase 8 - Validation Report
============================
Reads : complaints_classified.csv, topic_explanations.csv
Writes: validation_report.txt  (full report saved to disk)

Sections
--------
1.  Corpus overview & final topic sizes
2.  Cluster coherence (c-TF-IDF signal strength per topic)
3.  Per-topic spot check  -- 10 random task names per cluster
4.  BERTopic vs existing Asana 'Issue type' cross-tabulation
5.  Outlier analysis      -- 20 sampled tasks from topic -1
6.  Validation checklist  -- go/no-go before Phase 9
"""

import sys
import random
from collections import Counter
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
CLASSIFIED_FILE = Path("complaints_classified.csv")
EXPLAIN_FILE    = Path("topic_explanations.csv")
OUTPUT_FILE     = Path("validation_report.txt")

SAMPLE_N    = 10    # task names shown per topic in spot-check
OUTLIER_N   = 20    # task names shown from the outlier bin
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
for f in [CLASSIFIED_FILE, EXPLAIN_FILE]:
    if not f.exists():
        sys.exit("ERROR: '{}' not found. Run model_pipeline.py first.".format(f))

print("Loading {} ...".format(CLASSIFIED_FILE))
df  = pd.read_csv(CLASSIFIED_FILE, dtype=str, keep_default_na=False)
exp = pd.read_csv(EXPLAIN_FILE,    dtype=str, keep_default_na=False)

df["topic_id"]  = pd.to_numeric(df["topic_id"],  errors="coerce").fillna(-1).astype(int)
exp["topic_id"] = pd.to_numeric(exp["topic_id"], errors="coerce").fillna(-1).astype(int)
exp["topic_size"]   = pd.to_numeric(exp["topic_size"],  errors="coerce").fillna(0).astype(int)
exp["ctfidf_01_f"]  = pd.to_numeric(exp["ctfidf_01"],   errors="coerce").fillna(0.0)

total        = len(df)
n_classified = int((df["topic_id"] != -1).sum())
n_outliers   = int((df["topic_id"] == -1).sum())

# Build quick lookup: topic_id -> business_label (from classified CSV)
def get_blabel(tid):
    rows = df[df["topic_id"] == tid]["business_label"]
    return rows.iloc[0].strip() if len(rows) > 0 else "(no label)"

# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
lines = []

def out(s=""):
    lines.append(s)
    print(s)

W   = 72
SEP = "=" * W
S2  = "-" * W

def section(title):
    out()
    out(SEP)
    out("  " + title)
    out(SEP)

def coherence_signal(score):
    if score >= 0.10:
        return "STRONG    (very tight cluster)"
    elif score >= 0.05:
        return "GOOD      (clearly defined)"
    elif score >= 0.02:
        return "MODERATE  (some vocabulary overlap)"
    else:
        return "WEAK      (diffuse / catch-all)"

# --------------------------------------------------------------------------
# SECTION 1 - Corpus Overview
# --------------------------------------------------------------------------
section("1. CORPUS OVERVIEW")

out()
out("  Total tasks in corpus   : {:,}".format(total))
out("  Assigned to a topic     : {:,}  ({:.1f}%)".format(n_classified, n_classified / total * 100))
out("  Outliers  (topic -1)    : {:,}  ({:.1f}%)".format(n_outliers,   n_outliers   / total * 100))
out()
out("  {:<5}  {:>6}  {:>7}  {}".format("ID", "Size", "% Total", "Business Label"))
out("  " + S2)

topic_rows_sorted = (
    exp[exp["topic_id"] != -1]
    .sort_values("topic_size", ascending=False)
)
for _, row in topic_rows_sorted.iterrows():
    tid  = int(row["topic_id"])
    size = int(row["topic_size"])
    pct  = size / total * 100
    out("  {:<5}  {:>6,}  {:>6.1f}%  {}".format(tid, size, pct, get_blabel(tid)))

out("  {:<5}  {:>6,}  {:>6.1f}%  Outliers / Unclassified".format(
    -1, n_outliers, n_outliers / total * 100))

# --------------------------------------------------------------------------
# SECTION 2 - Cluster Coherence
# --------------------------------------------------------------------------
section("2. CLUSTER COHERENCE  (c-TF-IDF signal strength)")

out()
out("  The top c-TF-IDF score measures how uniquely the leading keyword")
out("  identifies a cluster vs the rest of the corpus.")
out("  Higher score = tighter, more semantically consistent cluster.")
out()
out("  {:<5}  {:>8}  {:<35}  {}".format("ID", "Score", "Signal", "Business Label"))
out("  " + S2)

for _, row in exp[exp["topic_id"] != -1].sort_values("ctfidf_01_f", ascending=False).iterrows():
    tid   = int(row["topic_id"])
    score = float(row["ctfidf_01_f"])
    out("  {:<5}  {:>8.4f}  {:<35}  {}".format(
        tid, score, coherence_signal(score), get_blabel(tid)))

out()
out("  Topics with WEAK signal are the most likely to contain mixed content.")
out("  Review their spot-checks (Section 3) most carefully.")

# --------------------------------------------------------------------------
# SECTION 3 - Per-Topic Spot Check
# --------------------------------------------------------------------------
section("3. PER-TOPIC SPOT CHECK  ({} random task names per cluster)".format(SAMPLE_N))

out()
out("  For each cluster, read the task names and ask:")
out("    - Do these tasks share the same type of work?")
out("    - Are there any obvious mis-assignments?")
out("    - Does the business label accurately describe the cluster?")

non_outlier_ids = sorted(df[df["topic_id"] != -1]["topic_id"].unique())

for tid in non_outlier_ids:
    subset = df[df["topic_id"] == tid]
    blabel = get_blabel(tid)
    size   = len(subset)

    out()
    out("  +-Topic {:2d}-+-{}-+-{:,} tasks-+".format(tid, blabel, size))

    sample_rows = subset.sample(min(SAMPLE_N, size), random_state=RANDOM_SEED)
    for i, (_, row) in enumerate(sample_rows.iterrows(), 1):
        name      = str(row.get("Name", "")).strip()[:85]
        intent    = str(row.get("form_intent", "")).strip()[:30]
        intent_str = "  [{}]".format(intent) if intent else ""
        out("  {:>2}. {}{}".format(i, name if name else "(no name)", intent_str))

# --------------------------------------------------------------------------
# SECTION 4 - Cross-tabulation: BERTopic vs Asana Issue type
# --------------------------------------------------------------------------
section("4. BERTOPIC vs EXISTING ASANA 'Issue type' / 'Issue sub-type'")

out()

for tag_col in ["Issue type", "Issue sub-type"]:
    if tag_col not in df.columns:
        out("  NOTE: column '{}' not found in classified CSV. Skipping.".format(tag_col))
        continue

    tagged_df     = df[df[tag_col].str.strip() != ""].copy()
    n_tagged      = len(tagged_df)
    pct_tagged    = n_tagged / total * 100 if total else 0

    out("  -- {} --".format(tag_col))
    out("  Rows with '{}' filled: {:,} of {:,} ({:.1f}%)".format(
        tag_col, n_tagged, total, pct_tagged))
    out()

    if n_tagged == 0:
        out("  No filled rows found - skipping this column.")
        out()
        continue

    out("  For each BERTopic cluster, top existing '{}' labels:".format(tag_col))
    out()

    for tid in non_outlier_ids:
        subset = tagged_df[tagged_df["topic_id"] == tid]
        if subset.empty:
            continue
        blabel  = get_blabel(tid)
        counts  = Counter(subset[tag_col].str.strip())
        top5    = counts.most_common(5)
        n_in    = len(subset)

        out("  Topic {:2d} | {} | {:,} tagged rows".format(tid, blabel, n_in))
        for lbl, cnt in top5:
            pct = cnt / n_in * 100
            bar = "#" * max(1, int(pct / 4))
            out("    {:>5,}  {:>5.1f}%  {}  {}".format(cnt, pct, bar, lbl[:55]))
        out()

    # Overall alignment summary: top BERTopic label per Asana Issue type
    out("  Reverse view -- top BERTopic cluster per Asana '{}' label:".format(tag_col))
    out()
    all_labels = [l.strip() for l in tagged_df[tag_col] if l.strip()]
    top_asana  = Counter(all_labels).most_common(15)

    for asana_lbl, total_cnt in top_asana:
        subset   = tagged_df[tagged_df[tag_col].str.strip() == asana_lbl]
        top_bert = Counter(subset["business_label"].str.strip()).most_common(3)
        out("  '{}' ({:,} tasks)".format(asana_lbl[:55], total_cnt))
        for blbl, cnt in top_bert:
            pct = cnt / total_cnt * 100
            out("      -> {:.0f}%  {}".format(pct, blbl[:60]))
        out()

# --------------------------------------------------------------------------
# SECTION 5 - Outlier Analysis
# --------------------------------------------------------------------------
section("5. OUTLIER ANALYSIS  ({} sampled tasks from topic -1)".format(OUTLIER_N))

out()
out("  These {:,} tasks were not assigned to any cluster.".format(n_outliers))
out("  Look for hidden patterns: any group of 5+ similar tasks")
out("  might warrant a new seed topic and a model re-run.")
out()

outliers = df[df["topic_id"] == -1]
sample_out = outliers.sample(min(OUTLIER_N, len(outliers)), random_state=RANDOM_SEED)

for i, (_, row) in enumerate(sample_out.iterrows(), 1):
    name    = str(row.get("Name", "")).strip()[:80]
    intent  = str(row.get("form_intent", "")).strip()[:25]
    sec     = str(row.get("Section/Column", "")).strip()[:20]
    parts   = []
    if intent: parts.append("intent={}".format(intent))
    if sec:    parts.append("status={}".format(sec))
    meta = "  [{}]".format(", ".join(parts)) if parts else ""
    out("  {:>2}. {}{}".format(i, name if name else "(no name)", meta))

out()
out("  Full outlier list: filter complaints_classified.csv for topic_id == -1")
out("  (Total: {:,} rows)".format(n_outliers))

# --------------------------------------------------------------------------
# SECTION 6 - Validation Checklist
# --------------------------------------------------------------------------
section("6. VALIDATION CHECKLIST")

out()
out("  Work through each item. Check it off when satisfied.")
out()
out("  SPOT-CHECK (Section 3)")
out("  [ ] All tasks within each topic share a coherent operational theme")
out("  [ ] No obvious mis-assignments visible in the 10-task samples")
out("  [ ] Business labels in BUSINESS_LABELS dict match your intent")
out()
out("  CROSS-TABULATION (Section 4)")
out("  [ ] BERTopic topics broadly align with existing Asana Issue type labels")
out("  [ ] Any divergence is explainable (e.g. BERTopic is more granular)")
out("  [ ] No BERTopic topic is dominated by a single unexpected Asana label")
out()
out("  OUTLIERS (Section 5)")
out("  [ ] Outlier tasks are genuinely unique or edge-cases")
out("  [ ] No hidden sub-group of 10+ similar tasks visible in the sample")
out()
out("  COHERENCE (Section 2)")
out("  [ ] WEAK-signal topics reviewed and labels confirmed as accurate")
out("  [ ] Topic 16 (General Operations) accepted as intended catch-all")
out()
out("  TAXONOMY SIGN-OFF")
out("  [ ] Final 18 business labels reviewed and approved by domain expert")
out("  [ ] complaints_classified.csv ready to share with stakeholders")
out()
out("  IF ALL BOXES CHECKED -> taxonomy is ready. Proceed to Phase 9 (classifier)")
out("  IF ISSUES FOUND      -> note topic IDs and adjust BUSINESS_LABELS or seeds")

out()
out(SEP)
out("  VALIDATION COMPLETE")
out("  Saved -> {}".format(OUTPUT_FILE))
out(SEP)

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8-sig") as fh:
    fh.write("\n".join(lines))
