"""
add_dendrogram.py
==================
Generates the Ward's linkage hierarchical dendrogram and adds it
to the leadership_report.html as a new section.
Inserted after the SPLIT_VALIDATION section.
Does NOT modify any existing content.
"""
import base64
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from pathlib import Path

REPORT_PATH = Path("outputs/reports/leadership_report.html")
EMB_PATH = Path("outputs/embeddings.npy")
TRAIN_PATH = Path("outputs/training_data.csv")
CORPUS_PATH = Path("outputs/corpus_clean.csv")


def build_dendrogram_html():
    """Build interactive Plotly dendrogram from cluster centroids."""
    print("  Loading embeddings and labels...")
    embeddings = np.load(str(EMB_PATH))
    train_df = pd.read_csv(TRAIN_PATH, dtype=str, keep_default_na=False)
    corpus = pd.read_csv(CORPUS_PATH, dtype=str, keep_default_na=False)

    # Map task_id -> label
    label_map = dict(zip(train_df["task_id"], train_df["label"]))
    labels = [label_map.get(tid, "Unknown") for tid in corpus["task_id"]]

    # Get unique labels and compute centroids
    unique_labels = sorted(set(labels) - {"Unknown"})
    centroids = []
    for lbl in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == lbl]
        centroid = embeddings[indices].mean(axis=0)
        centroids.append(centroid)

    centroids_matrix = np.array(centroids)

    # Ward's linkage
    print("  Computing Ward's linkage dendrogram...")
    condensed_dist = pdist(centroids_matrix, metric='cosine')
    Z = linkage(condensed_dist, method='ward')

    # Use scipy dendrogram to get the tree structure
    from scipy.cluster.hierarchy import cophenet
    coph_corr, _ = cophenet(Z, condensed_dist)
    print(f"  Cophenetic correlation: {coph_corr:.4f}")

    # Short labels for display
    short_labels = [l[:25] for l in unique_labels]

    # Create Plotly dendrogram using figure_factory
    fig = ff.create_dendrogram(
        centroids_matrix,
        orientation='bottom',
        labels=short_labels,
        distfun=lambda x: pdist(x, metric='cosine'),
        linkagefun=lambda x: linkage(x, method='ward'),
        color_threshold=0.55,
    )

    # Add a horizontal line at the merge threshold
    fig.add_hline(y=0.2, line_dash="dash", line_color="red",
                  annotation_text="Merge threshold (d=0.2)",
                  annotation_position="top left")

    # Add another line showing the 8-group cut
    fig.add_hline(y=0.55, line_dash="dot", line_color="blue",
                  annotation_text="8 Issue Types cut",
                  annotation_position="top left")

    fig.update_layout(
        title=dict(
            text=f"Hierarchical Cluster Tree (Ward's Linkage, Cosine Distance) | Cophenetic r = {coph_corr:.3f}",
            font=dict(size=15)
        ),
        xaxis=dict(title="", tickfont=dict(size=9), tickangle=45),
        yaxis=dict(title="Ward's Distance (Cosine)"),
        height=550,
        width=1100,
        margin=dict(l=60, r=40, t=70, b=180),
        paper_bgcolor="white",
        plot_bgcolor="#fafcff",
    )

    return fig.to_html(full_html=False, include_plotlyjs=False), coph_corr


def main():
    print("=" * 60)
    print("ADD HIERARCHICAL DENDROGRAM TO DASHBOARD")
    print("=" * 60)

    dendrogram_html, coph_corr = build_dendrogram_html()

    new_section = f"""
<!-- DENDROGRAM_SECTION_START -->
<div class="section" style="border-top: 4px solid #6366f1; margin-top:40px;">
    <h2>Hierarchical Cluster Tree (Ward's Linkage)</h2>
    <p style="color:#64748b; margin-bottom:10px; font-size:13px;">
        Shows how similar clusters are to each other based on centroid cosine distance.
        Clusters that join at lower heights are more similar. Used to identify merge candidates
        (red dashed line, d&lt;0.2) and to determine the 8 Issue Type groupings (blue dotted line).
    </p>
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-bottom:16px;">
        <div style="background:#f8fafc; padding:10px; border-radius:6px; text-align:center;">
            <strong>Cophenetic r = {coph_corr:.3f}</strong><br>
            <span style="font-size:11px; color:#666;">How well the tree preserves pairwise distances</span>
        </div>
        <div style="background:#fef2f2; padding:10px; border-radius:6px; text-align:center;">
            <strong style="color:#dc2626;">Red line (d=0.2)</strong><br>
            <span style="font-size:11px; color:#666;">Merge candidate threshold</span>
        </div>
        <div style="background:#eff6ff; padding:10px; border-radius:6px; text-align:center;">
            <strong style="color:#2563eb;">Blue line (d=0.55)</strong><br>
            <span style="font-size:11px; color:#666;">Cut for 8 Issue Types</span>
        </div>
    </div>
    {dendrogram_html}
</div>
<!-- DENDROGRAM_SECTION_END -->
"""

    # Read existing report
    content = REPORT_PATH.read_text(encoding="utf-8")

    if "DENDROGRAM_SECTION_START" in content:
        start = content.find("<!-- DENDROGRAM_SECTION_START -->")
        end = content.find("<!-- DENDROGRAM_SECTION_END -->") + len("<!-- DENDROGRAM_SECTION_END -->")
        content = content[:start] + new_section + content[end:]
        print("  Replaced existing dendrogram section.")
    else:
        # Insert after SPLIT_VALIDATION_SECTION_END
        marker = "<!-- SPLIT_VALIDATION_SECTION_END -->"
        idx = content.find(marker)
        if idx < 0:
            # Fallback: insert before SUBTYPE_DEFINITIONS
            marker = "<!-- SUBTYPE_DEFINITIONS_SECTION_START -->"
            idx = content.find(marker)
        else:
            idx += len(marker)

        content = content[:idx] + "\n" + new_section + content[idx:]
        print("  Inserted dendrogram section after Split Validation.")

    REPORT_PATH.write_text(content, encoding="utf-8")
    size_kb = REPORT_PATH.stat().st_size / 1024
    print(f"\n  Report size: {size_kb:.0f} KB")
    print("  Done!")


if __name__ == "__main__":
    main()
