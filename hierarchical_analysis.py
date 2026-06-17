"""
Hierarchical Cluster Analysis: Dendrogram + Semantic Validation
================================================================
1. Compute centroid embeddings for each class
2. Generate dendrogram using Ward's linkage
3. Identify merge candidates (distance < 0.2)
4. Semantic overlap report (top 10 words per cluster pair)
5. Validation table (merge pair, cophenetic distance, cosine similarity)
6. Inject results into leadership_report.html
"""
import os, sys, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist, cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go
import plotly.figure_factory as ff

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

TRAINING_PATH = Path("outputs/training_data.csv")
EMBEDDINGS_PATH = Path("outputs/embeddings.npy")
CORPUS_PATH = Path("outputs/corpus_clean.csv")
REPORT_PATH = Path("outputs/reports/leadership_report.html")

MERGE_DISTANCE_THRESHOLD = 0.2


def main():
    print("=" * 70)
    print("HIERARCHICAL CLUSTER ANALYSIS")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data...")
    corpus = pd.read_csv(CORPUS_PATH, dtype=str, keep_default_na=False)
    train = pd.read_csv(TRAINING_PATH, dtype=str, keep_default_na=False)
    embeddings = np.load(str(EMBEDDINGS_PATH))

    label_map = dict(zip(train["task_id"], train["label"]))
    labels = [label_map.get(tid, "Unknown") for tid in corpus["task_id"]]
    texts = corpus["cleaned_text"].tolist()

    unique_labels = sorted(set(labels))
    print(f"  Classes: {len(unique_labels)}")

    # Compute centroids
    print("\n[2/5] Computing class centroids...")
    centroids = []
    label_names = []
    label_texts = {}
    for lbl in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == lbl]
        centroid = embeddings[indices].mean(axis=0)
        centroids.append(centroid)
        label_names.append(lbl)
        label_texts[lbl] = [texts[i] for i in indices]

    centroids_matrix = np.array(centroids)
    print(f"  Centroid matrix: {centroids_matrix.shape}")

    # Compute linkage (Ward's method)
    print("\n[3/5] Computing Ward's linkage + dendrogram...")
    # Use 1 - cosine_similarity as distance
    dist_matrix = 1 - cosine_similarity(centroids_matrix)
    # Convert to condensed form for linkage
    condensed_dist = pdist(centroids_matrix, metric='cosine')
    Z = linkage(condensed_dist, method='ward')

    # Cophenetic correlation
    coph_dist, coph_matrix = cophenet(Z, condensed_dist)
    print(f"  Cophenetic correlation: {coph_dist:.4f}")

    # Create dendrogram with Plotly
    short_labels = [l[:25] for l in label_names]

    # Use scipy dendrogram to get structure, then plot with plotly
    from scipy.cluster.hierarchy import dendrogram as scipy_dendro
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_mpl, ax = plt.subplots(1, 1, figsize=(14, 8))
    dendro_data = scipy_dendro(Z, labels=short_labels, ax=ax, leaf_rotation=90,
                                color_threshold=MERGE_DISTANCE_THRESHOLD)
    plt.close(fig_mpl)

    # Build Plotly dendrogram
    fig = ff.create_dendrogram(
        centroids_matrix,
        labels=short_labels,
        linkagefun=lambda x: linkage(x, method='ward'),
        distfun=lambda x: pdist(x, metric='cosine'),
    )
    fig.update_layout(
        title=dict(text="Taxonomy Dendrogram (Ward's Linkage, Cosine Distance)", font=dict(size=16)),
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(title="Distance"),
        height=550,
        margin=dict(l=60, r=40, t=60, b=180),
        paper_bgcolor="white",
        shapes=[dict(type="line", x0=-0.5, x1=len(unique_labels)-0.5,
                     y0=MERGE_DISTANCE_THRESHOLD, y1=MERGE_DISTANCE_THRESHOLD,
                     line=dict(color="red", width=2, dash="dash"))],
        annotations=[dict(x=len(unique_labels)-2, y=MERGE_DISTANCE_THRESHOLD+0.02,
                         text=f"Merge threshold ({MERGE_DISTANCE_THRESHOLD})",
                         showarrow=False, font=dict(color="red", size=11))]
    )
    dendro_html = fig.to_html(full_html=False, include_plotlyjs=False)

    # Identify merge candidates (pairs that merge below threshold)
    print(f"\n[4/5] Identifying merge candidates (distance < {MERGE_DISTANCE_THRESHOLD})...")
    n = len(unique_labels)
    merge_candidates = []

    # Check all pairs' cosine distance
    sim_matrix = cosine_similarity(centroids_matrix)
    for i in range(n):
        for j in range(i+1, n):
            dist = 1 - sim_matrix[i][j]
            if dist < MERGE_DISTANCE_THRESHOLD:
                # Get top 10 words for each
                vec_i = CountVectorizer(stop_words="english", max_features=10, min_df=1)
                vec_j = CountVectorizer(stop_words="english", max_features=10, min_df=1)
                try:
                    vec_i.fit(label_texts[label_names[i]][:100])
                    words_i = list(vec_i.vocabulary_.keys())
                except:
                    words_i = []
                try:
                    vec_j.fit(label_texts[label_names[j]][:100])
                    words_j = list(vec_j.vocabulary_.keys())
                except:
                    words_j = []

                overlap = set(words_i) & set(words_j)

                merge_candidates.append({
                    "class_a": label_names[i],
                    "class_b": label_names[j],
                    "cosine_distance": dist,
                    "cosine_similarity": sim_matrix[i][j],
                    "words_a": ", ".join(words_i[:8]),
                    "words_b": ", ".join(words_j[:8]),
                    "word_overlap": ", ".join(list(overlap)[:6]),
                    "overlap_pct": len(overlap) / max(len(set(words_i) | set(words_j)), 1) * 100,
                })

    merge_candidates.sort(key=lambda x: x["cosine_distance"])
    print(f"  Found {len(merge_candidates)} merge candidates")

    for mc in merge_candidates:
        print(f"    {mc['class_a'][:30]:<32} + {mc['class_b'][:30]:<32} dist={mc['cosine_distance']:.4f} sim={mc['cosine_similarity']:.4f}")

    # Build validation table
    print(f"\n[5/5] Generating report...")

    # Similarity heatmap (full matrix)
    fig_heat = go.Figure(data=go.Heatmap(
        z=sim_matrix,
        x=short_labels,
        y=short_labels,
        colorscale="RdYlGn",
        zmin=0.5, zmax=1.0,
        hovertemplate="%{y} vs %{x}<br>Similarity: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Cosine Sim"),
    ))
    fig_heat.update_layout(
        title=dict(text="Class Centroid Similarity Matrix", font=dict(size=16)),
        height=650, width=850,
        margin=dict(l=180, r=40, t=60, b=180),
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
        paper_bgcolor="white",
    )
    heatmap_html = fig_heat.to_html(full_html=False, include_plotlyjs=False)

    # Build HTML section
    section_html = f"""
<!-- HIERARCHICAL_ANALYSIS_START -->
<div class="section" style="border-top: 4px solid #911eb4;">
    <h2>Hierarchical Cluster Analysis &amp; Merge Validation</h2>
    <p style="color:#64748b; font-size:13px; margin-bottom:15px;">
        Ward's linkage dendrogram + cosine similarity validation. Red dashed line = merge threshold ({MERGE_DISTANCE_THRESHOLD}).
        Classes merging below this line are candidates for consolidation.
        Cophenetic correlation: <strong>{coph_dist:.4f}</strong>
    </p>

    <h3>Dendrogram (Ward's Method)</h3>
    {dendro_html}

    <h3 style="margin-top:30px;">Class Similarity Heatmap</h3>
    {heatmap_html}

    <h3 style="margin-top:30px;">Merge Validation Table</h3>
"""

    if merge_candidates:
        section_html += """<table style="width:100%; border-collapse:collapse; font-size:12px; margin:10px 0;">
<tr style="background:#2c3e50; color:white;">
<th style="padding:8px;">Class A</th><th style="padding:8px;">Class B</th>
<th style="padding:8px;">Cosine Distance</th><th style="padding:8px;">Similarity</th>
<th style="padding:8px;">Top Words A</th><th style="padding:8px;">Top Words B</th>
<th style="padding:8px;">Word Overlap</th></tr>"""
        for mc in merge_candidates:
            section_html += f"""<tr>
<td style="padding:6px; border-bottom:1px solid #e2e8f0;">{mc['class_a'][:30]}</td>
<td style="padding:6px; border-bottom:1px solid #e2e8f0;">{mc['class_b'][:30]}</td>
<td style="padding:6px; border-bottom:1px solid #e2e8f0; text-align:center;">{mc['cosine_distance']:.4f}</td>
<td style="padding:6px; border-bottom:1px solid #e2e8f0; text-align:center;">{mc['cosine_similarity']:.4f}</td>
<td style="padding:6px; border-bottom:1px solid #e2e8f0; font-size:11px;">{mc['words_a']}</td>
<td style="padding:6px; border-bottom:1px solid #e2e8f0; font-size:11px;">{mc['words_b']}</td>
<td style="padding:6px; border-bottom:1px solid #e2e8f0; font-size:11px;">{mc['word_overlap']}</td></tr>"""
        section_html += "</table>"
    else:
        section_html += """<p style="color:#16a34a; font-weight:bold;">No merge candidates found below distance threshold. All classes are well-separated.</p>"""

    section_html += """
    <div style="background:#fffbeb; border:1px solid #fbbf24; border-radius:6px; padding:12px; margin-top:15px;">
        <p style="font-size:12px; color:#78350f; margin:0;">
            <strong>Interpretation:</strong> Classes that appear close in the dendrogram AND have high word overlap
            are strong merge candidates. Classes with high similarity but low word overlap may share
            structural patterns but represent different business outcomes (keep separate).
        </p>
    </div>
</div>
<!-- HIERARCHICAL_ANALYSIS_END -->
"""

    # Inject into report
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    # Remove previous
    start_m = "<!-- HIERARCHICAL_ANALYSIS_START -->"
    end_m = "<!-- HIERARCHICAL_ANALYSIS_END -->"
    if start_m in html:
        html = html[:html.index(start_m)] + html[html.index(end_m) + len(end_m):]

    # Insert before taxonomy refinement or split validation
    for marker in ["<!-- TAXONOMY_REFINEMENT_START -->", "<!-- SPLIT_VALIDATION_START -->", '<div class="footer">']:
        if marker in html:
            pos = html.index(marker)
            break
    else:
        pos = html.rfind("</body>")

    html = html[:pos] + section_html + "\n" + html[pos:]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  Injected hierarchical analysis into {REPORT_PATH}")
    print(f"  Report size: {REPORT_PATH.stat().st_size / 1024:.0f} KB")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
