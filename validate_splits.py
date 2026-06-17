"""
Split Validation Analysis
==========================
For each class in training_data.csv:
1. Run KMeans(k=2) within the cluster
2. Calculate silhouette score (>0.3 = bimodal = split candidate)
3. Extract sub-cluster keywords + sample docs
4. Generate per-cluster UMAP visualization
5. Append results as new section to leadership_report.html

Only shows results where a split IS detected (silhouette > 0.3).
User reviews and decides whether to approve splits.
"""
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

SPLIT_THRESHOLD = 0.3  # Only show splits above this silhouette score
REPORT_PATH = Path("outputs/reports/leadership_report.html")
TRAINING_PATH = Path("outputs/training_data.csv")
EMBEDDINGS_PATH = Path("outputs/embeddings.npy")
CORPUS_PATH = Path("outputs/corpus_clean.csv")


def analyze_split(label, indices, embeddings, texts, task_ids):
    """Analyze whether a cluster should be split into 2."""
    if len(indices) < 40:
        return None  # Too small to split meaningfully

    cluster_embs = embeddings[indices]
    cluster_texts = [texts[i] for i in indices]
    cluster_ids = [task_ids[i] for i in indices]

    # KMeans k=2
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    sub_labels = km.fit_predict(cluster_embs)

    # Silhouette score
    sil = silhouette_score(cluster_embs, sub_labels, metric="cosine")

    if sil < SPLIT_THRESHOLD:
        return None  # Not bimodal enough

    # Extract keywords per sub-cluster
    sub_data = {}
    for sub_id in [0, 1]:
        sub_indices = [i for i in range(len(sub_labels)) if sub_labels[i] == sub_id]
        sub_texts = [cluster_texts[i] for i in sub_indices]
        sub_ids = [cluster_ids[i] for i in sub_indices]

        # Keywords via TF-IDF
        vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8, min_df=2)
        try:
            vec.fit(sub_texts)
            keywords = list(vec.vocabulary_.keys())[:8]
        except ValueError:
            keywords = []

        # Sample docs (5)
        samples = []
        for i in sub_indices[:5]:
            samples.append({"task_id": cluster_ids[i], "text": cluster_texts[i][:120]})

        sub_data[sub_id] = {
            "count": len(sub_indices),
            "keywords": keywords,
            "samples": samples,
        }

    # UMAP 2D for this cluster only
    from umap import UMAP as UMAPModel
    umap_mini = UMAPModel(n_neighbors=min(15, len(cluster_embs) - 1),
                          n_components=2, min_dist=0.1, metric="cosine", random_state=42)
    coords = umap_mini.fit_transform(cluster_embs)

    # Build mini scatter plot
    colors = ["#e6194b" if l == 0 else "#3cb44b" for l in sub_labels]
    fig = go.Figure()
    for sub_id, color, name in [(0, "#e6194b", "Sub-A"), (1, "#3cb44b", "Sub-B")]:
        mask = sub_labels == sub_id
        fig.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers",
            marker=dict(size=6, color=color, opacity=0.7),
            name=f"{name} ({sub_data[sub_id]['count']} docs)",
            hovertemplate=f"{name}<extra></extra>",
        ))
    # Centroids
    for sub_id, color, name in [(0, "#e6194b", "A"), (1, "#3cb44b", "B")]:
        mask = sub_labels == sub_id
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers",
            marker=dict(size=18, color=color, symbol="diamond", line=dict(color="black", width=2)),
            name=f"Centroid {name}", showlegend=False,
        ))

    fig.update_layout(
        title=dict(text=f"Split Analysis: {label[:40]}", font=dict(size=14)),
        xaxis=dict(showgrid=False, zeroline=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, title=""),
        height=350, width=500,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        legend=dict(font=dict(size=10)),
    )
    chart_html = fig.to_html(full_html=False, include_plotlyjs=False)

    return {
        "label": label,
        "total_docs": len(indices),
        "silhouette": sil,
        "sub_a": sub_data[0],
        "sub_b": sub_data[1],
        "chart_html": chart_html,
    }


def generate_split_section(results):
    """Generate HTML section for split candidates."""
    html = """
<div class="section" style="border-top: 4px solid #e6194b;">
    <h2>SPLIT VALIDATION REPORT</h2>
    <p style="color:#64748b; font-size:13px;">
        Only clusters with <strong>silhouette score > 0.3</strong> are shown below (bimodal clusters).
        Review each split proposal and decide whether it improves classification accuracy.
        <br><strong>Red</strong> = Sub-cluster A, <strong>Green</strong> = Sub-cluster B.
    </p>
"""

    if not results:
        html += """<p style="color:#52c41a; font-size:16px; font-weight:bold;">
            No splits detected. All clusters are unimodal (single-center). Taxonomy is stable.
        </p>"""
    else:
        html += f"<p><strong>{len(results)} split candidate(s) found:</strong></p>"

        for r in sorted(results, key=lambda x: -x["silhouette"]):
            html += f"""
<div style="background:#fff; border:1px solid #ddd; border-radius:8px; padding:20px; margin:15px 0; box-shadow:0 2px 4px rgba(0,0,0,0.05);">
    <h3 style="color:#1e293b;">{r['label']} ({r['total_docs']} docs)</h3>
    <p><strong>Silhouette Score: <span style="color:{'#e6194b' if r['silhouette']>0.4 else '#f58231'};">{r['silhouette']:.3f}</span></strong>
       {'(STRONG split signal)' if r['silhouette'] > 0.4 else '(Moderate split signal)'}</p>

    <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin:15px 0;">
        <div style="background:#fef2f2; border-radius:6px; padding:15px;">
            <h4 style="color:#e6194b;">Sub-cluster A ({r['sub_a']['count']} docs)</h4>
            <p><strong>Keywords:</strong> {', '.join(r['sub_a']['keywords'][:6])}</p>
            <p style="font-size:11px; color:#555;"><strong>Samples:</strong></p>
"""
            for s in r['sub_a']['samples'][:3]:
                html += f"<p style='font-size:11px; font-family:monospace; color:#666;'>[{s['task_id']}] {s['text'][:100]}...</p>"

            html += f"""
        </div>
        <div style="background:#f0fdf4; border-radius:6px; padding:15px;">
            <h4 style="color:#3cb44b;">Sub-cluster B ({r['sub_b']['count']} docs)</h4>
            <p><strong>Keywords:</strong> {', '.join(r['sub_b']['keywords'][:6])}</p>
            <p style="font-size:11px; color:#555;"><strong>Samples:</strong></p>
"""
            for s in r['sub_b']['samples'][:3]:
                html += f"<p style='font-size:11px; font-family:monospace; color:#666;'>[{s['task_id']}] {s['text'][:100]}...</p>"

            html += """
        </div>
    </div>
"""
            html += f"<div style='margin-top:10px;'>{r['chart_html']}</div>"
            html += """
    <p style="margin-top:15px; padding:10px; background:#fff3cd; border-radius:4px; font-size:12px;">
        <strong>YOUR DECISION:</strong> Should this cluster be split into two separate classes?
        If yes, suggest names for Sub-A and Sub-B.
    </p>
</div>
"""

    html += "</div>"
    return html


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 60)
    print("SPLIT VALIDATION ANALYSIS")
    print(f"Threshold: silhouette > {SPLIT_THRESHOLD}")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    # Load
    print("\n[1/3] Loading data...")
    corpus = pd.read_csv(CORPUS_PATH, dtype=str, keep_default_na=False)
    train = pd.read_csv(TRAINING_PATH, dtype=str, keep_default_na=False)
    embeddings = np.load(str(EMBEDDINGS_PATH))

    texts = corpus["cleaned_text"].tolist()
    task_ids = corpus["task_id"].tolist()

    # Map labels via training data
    label_map = dict(zip(train["task_id"], train["label"]))
    labels = [label_map.get(tid, "Unknown") for tid in task_ids]

    unique_labels = sorted(set(labels))
    print(f"  Classes: {len(unique_labels)}")
    print(f"  Docs: {len(texts)}")

    # Analyze each class
    print(f"\n[2/3] Running split analysis on each class...")
    results = []
    for lbl in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == lbl]
        count = len(indices)

        result = analyze_split(lbl, indices, embeddings, texts, task_ids)
        if result:
            print(f"  SPLIT DETECTED: {lbl} (sil={result['silhouette']:.3f}, {count} docs)")
            print(f"    Sub-A ({result['sub_a']['count']}): {', '.join(result['sub_a']['keywords'][:4])}")
            print(f"    Sub-B ({result['sub_b']['count']}): {', '.join(result['sub_b']['keywords'][:4])}")
            results.append(result)
        else:
            print(f"  OK: {lbl} ({count} docs) - no split needed")

    # Generate HTML section
    print(f"\n[3/3] Generating split report section...")
    split_html = generate_split_section(results)

    # Inject into leadership_report.html (before the closing tags)
    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report = f.read()

        # Remove any previous split section
        marker_start = "<!-- SPLIT_VALIDATION_START -->"
        marker_end = "<!-- SPLIT_VALIDATION_END -->"
        if marker_start in report:
            before = report[:report.index(marker_start)]
            after = report[report.index(marker_end) + len(marker_end):]
            report = before + after

        # Insert before footer
        injection = f"\n{marker_start}\n{split_html}\n{marker_end}\n"
        insert_point = report.rfind('<div class="footer">')
        if insert_point > 0:
            report = report[:insert_point] + injection + report[insert_point:]
        else:
            report = report.replace("</body>", injection + "</body>")

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  Injected split analysis into {REPORT_PATH}")
    else:
        # Save standalone
        standalone_path = Path("outputs/reports/split_validation.html")
        with open(standalone_path, "w", encoding="utf-8") as f:
            f.write(f"<html><body>{split_html}</body></html>")
        print(f"  Saved standalone: {standalone_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {len(results)} split candidate(s) found")
    print(f"{'=' * 60}")
    if results:
        for r in sorted(results, key=lambda x: -x["silhouette"]):
            print(f"  {r['label']:<40} sil={r['silhouette']:.3f}  ({r['sub_a']['count']}+{r['sub_b']['count']} docs)")
        print(f"\n  Review in browser, then tell me which splits to approve.")
    else:
        print(f"  All classes are unimodal. No splits needed. Taxonomy is stable.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
