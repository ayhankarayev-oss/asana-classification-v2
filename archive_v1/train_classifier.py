"""
Phase 9 - Production Classifier
=================================
Reads : complaints_classified.csv
Writes: issue_classifier.pkl    (trained model -- use this to score new tasks)
        classifier_report.txt   (full evaluation report)

Why a separate classifier?
---------------------------
BERTopic takes ~3 minutes to run and requires the full corpus in memory.
This TF-IDF + Logistic Regression model loads in <1 second and classifies
a new task in milliseconds -- suitable for real-time or batch use.

Training signal:
The BERTopic-assigned 'business_label' column is used as ground truth.
Outlier rows (topic_id == -1) are excluded from training.

Usage after training:
    import joblib
    clf = joblib.load("issue_classifier.pkl")
    label      = clf.predict(["your task name + notes here"])[0]
    confidence = clf.predict_proba(["your task name + notes here"]).max()
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
CLASSIFIED_FILE  = Path("complaints_classified.csv")
MODEL_FILE       = Path("issue_classifier.pkl")
REPORT_FILE      = Path("classifier_report.txt")

TEST_SIZE        = 0.20    # 20% held-out test set
CV_FOLDS         = 5       # stratified k-fold cross-validation
RANDOM_STATE     = 42
DEMO_N           = 6       # number of example predictions shown in demo

# --------------------------------------------------------------------------
# Load labeled data
# --------------------------------------------------------------------------
if not CLASSIFIED_FILE.exists():
    sys.exit("ERROR: '{}' not found. Run model_pipeline.py first.".format(CLASSIFIED_FILE))

print("Loading {} ...".format(CLASSIFIED_FILE))
df = pd.read_csv(CLASSIFIED_FILE, dtype=str, keep_default_na=False)
df["topic_id"] = pd.to_numeric(df["topic_id"], errors="coerce").fillna(-1).astype(int)

# Keep only rows that were assigned a cluster (exclude outliers)
labeled = df[df["topic_id"] != -1].copy()
labeled = labeled[labeled["complaint_text"].str.strip() != ""]
labeled = labeled[labeled["business_label"].str.strip() != ""]

X = labeled["complaint_text"].tolist()
y = labeled["business_label"].tolist()

total      = len(df)
n_labeled  = len(labeled)
n_outliers = int((df["topic_id"] == -1).sum())
n_classes  = len(set(y))

print("  Total rows in CSV    : {:,}".format(total))
print("  Training rows        : {:,}  (excludes {:,} outliers)".format(n_labeled, n_outliers))
print("  Target classes       : {}".format(n_classes))

# --------------------------------------------------------------------------
# Build pipeline
# --------------------------------------------------------------------------
# TfidfVectorizer:
#   - ngram_range=(1,2)  : captures bigrams like "capital call", "cost basis"
#   - sublinear_tf=True  : log-scales term frequencies, helps long notes
#   - min_df=2           : ignore terms that appear in only 1 document
#   - max_features=25000 : cap vocabulary size for memory efficiency
#
# LogisticRegression:
#   - class_weight='balanced': compensates for small classes (51-898 docs/class)
#   - C=5                    : moderate regularization
#   - max_iter=1000          : enough iterations for convergence on this vocabulary
#   - solver='lbfgs'         : efficient for multi-class text classification
# --------------------------------------------------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range  = (1, 2),
        sublinear_tf = True,
        min_df       = 2,
        max_features = 25000,
        strip_accents = "unicode",
        analyzer     = "word",
        token_pattern= r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b",
    )),
    ("clf", LogisticRegression(
        class_weight = "balanced",
        C            = 5,
        max_iter     = 1000,
        solver       = "lbfgs",
        multi_class  = "multinomial",
        random_state = RANDOM_STATE,
    )),
])

# --------------------------------------------------------------------------
# Train / test split
# --------------------------------------------------------------------------
print("\nSplitting {:,} rows ({:.0f}% train / {:.0f}% test, stratified) ...".format(
    n_labeled, (1 - TEST_SIZE) * 100, TEST_SIZE * 100))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    stratify     = y,
    random_state = RANDOM_STATE,
)

print("  Train : {:,} rows".format(len(X_train)))
print("  Test  : {:,} rows".format(len(X_test)))

# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------
print("\nTraining TF-IDF + Logistic Regression pipeline ...")
pipeline.fit(X_train, y_train)
print("  Done.")

# --------------------------------------------------------------------------
# Evaluate on held-out test set
# --------------------------------------------------------------------------
print("\nEvaluating on test set ...")
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)

test_accuracy = float(np.mean(np.array(y_pred) == np.array(y_test)))
top1_conf     = float(y_proba.max(axis=1).mean())   # avg confidence of top prediction

print("  Test accuracy (top-1) : {:.1f}%".format(test_accuracy * 100))
print("  Avg confidence        : {:.3f}".format(top1_conf))

report_str = classification_report(y_test, y_pred, digits=3, zero_division=0)

# --------------------------------------------------------------------------
# 5-fold cross-validation on the full labeled set
# --------------------------------------------------------------------------
print("\nRunning {}-fold cross-validation on full labeled set ...".format(CV_FOLDS))
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=1)
print("  CV scores  : {}".format("  ".join("{:.3f}".format(s) for s in cv_scores)))
print("  CV mean    : {:.3f}  (+/- {:.3f})".format(cv_scores.mean(), cv_scores.std() * 2))

# --------------------------------------------------------------------------
# Retrain on full labeled set (now we have the CV estimate, use all data)
# --------------------------------------------------------------------------
print("\nRetraining on full labeled set ({:,} rows) for the saved model ...".format(n_labeled))
pipeline.fit(X, y)
print("  Done.")

# --------------------------------------------------------------------------
# Save model
# --------------------------------------------------------------------------
joblib.dump(pipeline, MODEL_FILE)
print("\nModel saved -> {}".format(MODEL_FILE))

# --------------------------------------------------------------------------
# Build text report
# --------------------------------------------------------------------------
lines = []

def ln(s=""):
    lines.append(s)

W   = 72
SEP = "=" * W
S2  = "-" * W

ln(SEP)
ln("  PHASE 9 - CLASSIFIER EVALUATION REPORT")
ln(SEP)
ln()
ln("  Model       : TF-IDF (ngram 1-2, max 25k features) + Logistic Regression")
ln("  Training    : {:,} labeled tasks (outliers excluded)".format(n_labeled))
ln("  Classes     : {}".format(n_classes))
ln("  Test split  : {:.0f}% / {:.0f}% (stratified)".format(
    (1 - TEST_SIZE) * 100, TEST_SIZE * 100))
ln()
ln(S2)
ln("  HELD-OUT TEST SET RESULTS  ({:,} rows)".format(len(X_test)))
ln(S2)
ln()
ln("  Accuracy (top-1)  : {:.1f}%".format(test_accuracy * 100))
ln("  Avg confidence    : {:.3f}".format(top1_conf))
ln()
ln("  Per-class breakdown:")
ln()
# Indent the sklearn report
for line in report_str.splitlines():
    ln("    " + line)

ln()
ln(S2)
ln("  {}-FOLD CROSS-VALIDATION  (full labeled set, {:,} rows)".format(CV_FOLDS, n_labeled))
ln(S2)
ln()
ln("  Fold scores : {}".format("  ".join("{:.3f}".format(s) for s in cv_scores)))
ln("  Mean        : {:.3f}".format(cv_scores.mean()))
ln("  Std x2      : {:.3f}".format(cv_scores.std() * 2))
ln("  95% CI      : [{:.3f} -- {:.3f}]".format(
    cv_scores.mean() - cv_scores.std() * 2,
    cv_scores.mean() + cv_scores.std() * 2))

# Class size breakdown
ln()
ln(S2)
ln("  CLASS SIZE BREAKDOWN  (training distribution)")
ln(S2)
ln()
ln("  {:<50} {:>6}  {:>7}".format("Business Label", "Docs", "% Total"))
ln("  " + S2)
from collections import Counter
label_counts = Counter(y)
for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
    ln("  {:<50} {:>6,}  {:>6.1f}%".format(lbl[:50], cnt, cnt / n_labeled * 100))

# Confidence distribution on test set
conf_vals = y_proba.max(axis=1)
ln()
ln(S2)
ln("  CONFIDENCE DISTRIBUTION ON TEST SET")
ln(S2)
ln()
ln("  Confidence bucket    Count    % of test")
ln("  " + S2)
buckets = [(0.9, 1.01, "0.90 - 1.00"), (0.7, 0.9, "0.70 - 0.90"),
           (0.5, 0.7,  "0.50 - 0.70"), (0.0, 0.5, "0.00 - 0.50")]
for lo, hi, label in buckets:
    n_bucket = int(((conf_vals >= lo) & (conf_vals < hi)).sum())
    pct = n_bucket / len(X_test) * 100
    bar = "#" * max(0, int(pct / 2))
    ln("  {:>18}  {:>6,}  {:>8.1f}%  {}".format(label, n_bucket, pct, bar))

# Instructions
ln()
ln(SEP)
ln("  HOW TO USE THE SAVED MODEL ON NEW TASKS")
ln(SEP)
ln()
ln("  # 1 - Classify a single new task")
ln("  import joblib")
ln("  clf = joblib.load('issue_classifier.pkl')")
ln("  complaint_text = 'Please add new private investment for Letsos in Addepar'")
ln("  label      = clf.predict([complaint_text])[0]")
ln("  confidence = clf.predict_proba([complaint_text]).max()")
ln("  print(label, confidence)")
ln()
ln("  # 2 - Score a batch of new tasks from a CSV")
ln("  import pandas as pd, joblib")
ln("  clf   = joblib.load('issue_classifier.pkl')")
ln("  new   = pd.read_csv('new_tasks.csv', dtype=str, keep_default_na=False)")
ln("  # Build complaint_text the same way preprocess.py does:")
ln("  new['complaint_text'] = (new['form_intent'].str.strip() + ' '")
ln("                          + new['Name'].str.strip() + ' '")
ln("                          + new['notes_clean'].str.strip()).str.strip()")
ln("  new['predicted_label']      = clf.predict(new['complaint_text'])")
ln("  new['prediction_confidence'] = clf.predict_proba(new['complaint_text']).max(axis=1)")
ln("  new.to_csv('new_tasks_classified.csv', index=False)")
ln()
ln("  # 3 - Confidence threshold: flag uncertain predictions for manual review")
ln("  uncertain = new[new['prediction_confidence'] < 0.60]")
ln("  print('{} tasks need manual review'.format(len(uncertain)))")
ln()
ln(SEP)
ln("  Model saved  -> {}".format(MODEL_FILE))
ln("  Report saved -> {}".format(REPORT_FILE))
ln(SEP)

report_text = "\n".join(lines)

# Print to console
print()
print(report_text)

# Save to file
with open(REPORT_FILE, "w", encoding="utf-8-sig") as fh:
    fh.write(report_text)

# --------------------------------------------------------------------------
# Live demo: predict DEMO_N random examples from the full dataset
# --------------------------------------------------------------------------
print()
print("=" * W)
print("  LIVE DEMO  -- {:} example predictions on held-out test rows".format(DEMO_N))
print("=" * W)
print()
print("  {:<70}  {:<8}  {}".format("Task Name (first 70 chars)", "Conf", "Predicted Label"))
print("  " + "-" * W)

import random
random.seed(RANDOM_STATE + 1)
demo_indices = random.sample(range(len(X_test)), min(DEMO_N, len(X_test)))

for idx in demo_indices:
    text   = X_test[idx]
    actual = y_test[idx]
    pred   = pipeline.predict([text])[0]
    conf   = float(pipeline.predict_proba([text]).max())
    # Show task Name (first 70 chars of complaint_text)
    short  = text[:70].replace("\n", " ")
    match  = "OK" if pred == actual else "!!"
    print("  {:<70}  {:.3f}   {}  {}".format(short, conf, match, pred[:55]))

print()
print("  OK = correct   !! = incorrect")
print()
print("Classifier pipeline complete.")
print("  Model -> {}".format(MODEL_FILE))
print("  Report -> {}".format(REPORT_FILE))