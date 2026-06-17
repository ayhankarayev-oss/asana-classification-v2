"""
predict.py
===========
Prediction module for the Asana task classifier.
Takes raw task text -> preprocesses -> embeds -> classifies.

Usage:
    from predict import TaskClassifier
    clf = TaskClassifier()
    result = clf.predict("please set up new account for client xyz")
    # result = {"sub_type": "Account Setup & Connectivity", "issue_type": "Account & Data Onboarding", "confidence": 0.87, "alternatives": [...]}
"""
import json
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_DIR = Path(__file__).parent / "models"


class TaskClassifier:
    def __init__(self):
        """Load model, label encoder, and embedding model."""
        # Load classifier
        with open(MODEL_DIR / "classifier_v2.pkl", "rb") as f:
            artifacts = pickle.load(f)
        self.model = artifacts["model"]
        self.label_encoder = artifacts["label_encoder"]

        # Load label -> pillar mapping
        with open(MODEL_DIR / "label_map_v2.json", "r") as f:
            self.label_to_pillar = json.load(f)

        # Load sentence transformer (same one used for training)
        print("Loading embedding model...")
        self.embedder = SentenceTransformer("all-mpnet-base-v2")
        print("Classifier ready.")

    def preprocess(self, text: str) -> str:
        """Basic preprocessing: lowercase, strip whitespace."""
        text = text.strip().lower()
        # Remove excess whitespace
        text = " ".join(text.split())
        return text

    def predict(self, text: str) -> dict:
        """
        Predict sub-type and Issue Type for a task description.
        
        Returns:
            dict with keys: sub_type, issue_type, confidence, alternatives
        """
        # Preprocess
        clean_text = self.preprocess(text)

        if not clean_text or len(clean_text) < 5:
            return {
                "sub_type": "Unknown",
                "issue_type": "Unknown",
                "confidence": 0.0,
                "alternatives": [],
            }

        # Embed
        embedding = self.embedder.encode([clean_text])

        # Predict probabilities
        proba = self.model.predict_proba(embedding)[0]
        classes = self.label_encoder.classes_

        # Top prediction
        top_idx = np.argmax(proba)
        sub_type = classes[top_idx]
        confidence = float(proba[top_idx])
        issue_type = self.label_to_pillar.get(sub_type, "Unknown")

        # Top 3 alternatives
        sorted_indices = np.argsort(proba)[::-1]
        alternatives = []
        for idx in sorted_indices[1:4]:  # top 2-4
            alt_label = classes[idx]
            alt_conf = float(proba[idx])
            alt_pillar = self.label_to_pillar.get(alt_label, "Unknown")
            alternatives.append({
                "sub_type": alt_label,
                "issue_type": alt_pillar,
                "confidence": alt_conf,
            })

        return {
            "sub_type": sub_type,
            "issue_type": issue_type,
            "confidence": confidence,
            "alternatives": alternatives,
        }

    def predict_batch(self, texts: list) -> list:
        """Predict for multiple texts at once (faster than one-by-one)."""
        clean_texts = [self.preprocess(t) for t in texts]
        embeddings = self.embedder.encode(clean_texts, show_progress_bar=False)
        probas = self.model.predict_proba(embeddings)
        classes = self.label_encoder.classes_

        results = []
        for proba in probas:
            top_idx = np.argmax(proba)
            sub_type = classes[top_idx]
            results.append({
                "sub_type": sub_type,
                "issue_type": self.label_to_pillar.get(sub_type, "Unknown"),
                "confidence": float(proba[top_idx]),
            })
        return results


# Quick test if run directly
if __name__ == "__main__":
    clf = TaskClassifier()

    test_texts = [
        "please set up new account for client and connect bank feed",
        "update the quarterly valuation for private investment",
        "run and deliver monthly billing report",
        "fix missing cost basis for the household",
        "grant access to new team member for addepar",
    ]

    print("\n" + "=" * 60)
    print("PREDICTION TEST")
    print("=" * 60)
    for text in test_texts:
        result = clf.predict(text)
        print(f"\nInput: \"{text[:60]}...\"")
        print(f"  Sub-type:   {result['sub_type']}")
        print(f"  Issue Type: {result['issue_type']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        if result['alternatives']:
            alt = result['alternatives'][0]
            print(f"  Alt #2:     {alt['sub_type']} ({alt['confidence']:.1%})")
