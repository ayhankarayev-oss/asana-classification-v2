"""
train_classifier.py
====================
Trains a Logistic Regression classifier on sentence-transformer embeddings
for the v2 taxonomy (19 sub-types, 8 Issue Types).

Outputs:
- models/classifier_v2.pkl (trained model)
- models/label_map_v2.json (sub-type -> Issue Type mapping)
- models/eval_report.txt (cross-validation metrics)
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

TRAIN_PATH = Path("outputs/training_data_v2.csv")
EMB_PATH = Path("outputs/embeddings.npy")
CORPUS_PATH = Path("outputs/corpus_clean.csv")
MODEL_DIR = Path("models")


def main():
    print("=" * 60)
    print("TRAIN CLASSIFIER (v2: 19 sub-types × 8 Issue Types)")
    print("=" * 60)

    MODEL_DIR.mkdir(exist_ok=True)

    # Load data
    print("\n[1/4] Loading data...")
    train_df = pd.read_csv(TRAIN_PATH, dtype=str, keep_default_na=False)
    corpus = pd.read_csv(CORPUS_PATH, dtype=str, keep_default_na=False)
    embeddings = np.load(str(EMB_PATH))

    # Align embeddings with training labels via task_id
    label_map = dict(zip(train_df["task_id"], train_df["label"]))
    pillar_map = dict(zip(train_df["task_id"], train_df["business_pillar"]))

    # Build aligned X, y arrays
    task_ids = corpus["task_id"].tolist()
    X_list, y_list = [], []
    for i, tid in enumerate(task_ids):
        if tid in label_map:
            X_list.append(embeddings[i])
            y_list.append(label_map[tid])

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"  Training samples: {len(X)}")
    print(f"  Embedding dim: {X.shape[1]}")
    print(f"  Classes: {len(set(y))}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train Logistic Regression
    print("\n[2/4] Training Logistic Regression...")
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y_encoded)
    print("  Training complete.")

    # Cross-validation evaluation
    print("\n[3/4] Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(clf, X, y_encoded, cv=cv)

    accuracy = accuracy_score(y_encoded, y_pred_cv)
    report = classification_report(y, le.inverse_transform(y_pred_cv), digits=3)

    print(f"\n  Cross-Validation Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"\n{report}")

    # Save evaluation report
    eval_text = f"""CLASSIFIER EVALUATION REPORT (v2)
{'='*50}
Model: Logistic Regression (C=1.0, multinomial)
Features: all-mpnet-base-v2 embeddings (768-dim)
Training samples: {len(X)}
Classes: {len(set(y))} sub-types, 8 Issue Types
Cross-validation: 5-fold stratified

ACCURACY: {accuracy:.4f} ({accuracy*100:.1f}%)

CLASSIFICATION REPORT:
{report}
"""
    (MODEL_DIR / "eval_report.txt").write_text(eval_text)

    # Save model
    print("\n[4/4] Saving model artifacts...")
    with open(MODEL_DIR / "classifier_v2.pkl", "wb") as f:
        pickle.dump({"model": clf, "label_encoder": le}, f)
    print(f"  Saved: models/classifier_v2.pkl")

    # Save label -> pillar mapping
    label_to_pillar = {}
    for lbl in sorted(set(y)):
        mask = train_df["label"] == lbl
        if mask.any():
            label_to_pillar[lbl] = train_df.loc[mask, "business_pillar"].iloc[0]

    with open(MODEL_DIR / "label_map_v2.json", "w") as f:
        json.dump(label_to_pillar, f, indent=2)
    print(f"  Saved: models/label_map_v2.json")

    print(f"  Saved: models/eval_report.txt")
    print(f"\n{'='*60}")
    print(f"DONE — Accuracy: {accuracy*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
