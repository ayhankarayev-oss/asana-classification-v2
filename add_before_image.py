"""
add_before_image.py - SIMPLIFIED
=================================
Extracts the Section 1 Plotly chart JSON from leadership_report.html,
renders it as a static PNG, and embeds it ABOVE the dynamic chart
as a "BEFORE" reference snapshot.
"""
import base64
import json
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import from_json
from pathlib import Path

REPORT_PATH = Path("outputs/reports/leadership_report.html")


def extract_first_plotly_json(html_content):
    """Extract the first Plotly figure JSON from the HTML."""
    # Plotly embeds data as: Plotly.newPlot("div-id", data, layout, config)
    # Find the first occurrence
    pattern = r'Plotly\.newPlot\(\s*"[^"]+"\s*,\s*(\[.*?\])\s*,\s*(\{.*?\})\s*,\s*\{' 
    # Actually Plotly uses a different format in to_html - let's find the JSON data block
    # The format is: var data = [...]; var layout = {...};
    # Or it could be inline in Plotly.react()
    
    # Look for the plotly data/layout pattern
    # In plotly to_html(include_plotlyjs=False), it produces script blocks like:
    # Plotly.newPlot("uuid", [{...traces...}], {...layout...}, {responsive: true})
    
    # Find all script blocks
    scripts = re.findall(r'<script>(.*?)</script>', html_content, re.DOTALL)
    
    for script in scripts:
        if 'Plotly' in script and 'Cluster Map' in script:
            # This is our scatter plot script - it's the first one with "Cluster Map"
            return script
    return None


def main():
    print("=" * 60)
    print("ADD BEFORE IMAGE (static snapshot of Section 1 chart)")
    print("=" * 60)

    content = REPORT_PATH.read_text(encoding="utf-8")
    
    # Instead of trying to re-render, let's use Plotly's built-in static export
    # by reconstructing the figure from Section 1.
    # 
    # Actually the SIMPLEST approach: use the same UMAP + labels that Section 1 uses
    # (which is the training_data.csv BEFORE our name fix, i.e. with old names)
    # But training_data already has new names...
    #
    # SIMPLEST OF ALL: just take a screenshot using kaleido from the existing 
    # Section 1 data. The Section 1 chart in the HTML uses the OLD labels 
    # (it was generated before our name fixes). Let me extract the figure data.
    
    from umap import UMAP as UMAPModel
    
    # Load data - use the same approach as Section 1 (build_cluster_scatter)
    # but with the OLD names (pre-rename). The Section 1 chart was generated
    # from training_data.csv when it still had the old names.
    # We need to recreate that: 21 classes including "Asset Transfers & Movements"
    
    corpus = pd.read_csv("outputs/corpus_clean.csv", dtype=str, keep_default_na=False)
    embeddings = np.load("outputs/embeddings.npy")
    
    # Reconstruct the 21-class labels using apply_merges NAME_MAP (before T22 redistribution)
    # This is what Section 1 shows
    NAME_MAP = {
        0: "New Account Onboarding",
        1: "New Private Investment Setup",
        2: "Invoicing & Billing",
        3: "Commitment & Capital Call Tracking",
        4: "Access & Permission Requests",
        5: "Contact & User Updates",
        6: "Missing Data & Attributes",
        7: "Client Report & View Management",
        8: "Security & Position Updates",
        9: "Ownership & Trust Structure",
        10: "Scheduled Valuation Feeds",
        11: "Recurring Data Maintenance & Audits",
        12: "Debt & Lending Instruments",
        13: "Position Cleanup & Deduplication",
        14: "Investment Performance Analysis",
        15: "Historical Data Backfill",
        16: "Distribution & Income Recording",
        17: "Client Report & View Management",  # merged into T7
        18: "Missing Data & Attributes",
        19: "Meetings, Reminders & Follow-ups",
        20: "Unclassified (Topic 20)",
        21: "Cost Basis Reconciliation",
        22: "Asset Transfers & Movements",
    }
    
    # Get topic assignments from discovery_medium.csv
    disc = pd.read_csv("outputs/discovery_medium.csv", dtype=str, keep_default_na=False)
    tid_to_topic = dict(zip(disc["task_id"], disc["topic_id"].astype(int)))
    
    # Apply merge (T17 -> T7) to get 21 classes (with T22 still present)
    labels = []
    for task_id in corpus["task_id"]:
        topic = tid_to_topic.get(task_id, -1)
        if topic == 17:
            topic = 7  # merged
        if topic == 18:
            topic = 6  # merged (Missing Data)
        labels.append(NAME_MAP.get(topic, "Unknown"))
    
    # 2D UMAP (same params as the original)
    print("  Computing 2D UMAP...")
    umap_2d = UMAPModel(n_neighbors=15, n_components=2, min_dist=0.1,
                        metric="cosine", random_state=42)
    coords_2d = umap_2d.fit_transform(embeddings)
    
    # Build the plot (matching the screenshot style)
    DISTINCT_COLORS = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
        "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
        "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#e6beff",
        "#ffe119", "#fabebe",
    ]
    
    unique_labels = sorted(set(labels))
    if "Unknown" in unique_labels:
        unique_labels.remove("Unknown")
    label_to_color = {l: DISTINCT_COLORS[i % len(DISTINCT_COLORS)] for i, l in enumerate(unique_labels)}
    
    plot_df = pd.DataFrame({"x": coords_2d[:, 0], "y": coords_2d[:, 1], "label": labels})
    plot_df = plot_df[plot_df["label"] != "Unknown"]
    
    centroids = plot_df.groupby("label").agg({"x": "mean", "y": "mean"}).reset_index()
    centroids["count"] = [len(plot_df[plot_df["label"] == lbl]) for lbl in centroids["label"]]
    
    fig = go.Figure()
    
    for lbl in unique_labels:
        mask = plot_df["label"] == lbl
        subset = plot_df[mask]
        count = len(subset)
        fig.add_trace(go.Scatter(
            x=subset["x"], y=subset["y"],
            mode="markers",
            marker=dict(size=4, color=label_to_color[lbl], opacity=0.5),
            name=f"{lbl[:28]} ({count})",
            showlegend=True,
            hovertemplate=f"<b>{lbl[:28]}</b><extra></extra>",
        ))
    
    # Centroid diamonds
    fig.add_trace(go.Scatter(
        x=centroids["x"], y=centroids["y"],
        mode="markers+text",
        marker=dict(size=14, color=[label_to_color[l] for l in centroids["label"]],
                    symbol="diamond", line=dict(color="black", width=1.5)),
        text=[f"{l[:16]}" for l in centroids["label"]],
        textposition="top center",
        textfont=dict(size=7, color="#333"),
        name="Centroids",
        showlegend=True,
    ))
    
    fig.update_layout(
        title=dict(text=f"Cluster Map ({len(unique_labels)} Classes, Merged)", font=dict(size=16)),
        xaxis=dict(title="UMAP-1", showgrid=True, gridcolor="#eee"),
        yaxis=dict(title="UMAP-2", showgrid=True, gridcolor="#eee"),
        height=700,
        width=1150,
        margin=dict(l=50, r=270, t=50, b=50),
        paper_bgcolor="white",
        plot_bgcolor="#fafcff",
        legend=dict(title="Classes (click to toggle)", font=dict(size=8), 
                    yanchor="top", y=1, xanchor="left", x=1.01),
    )
    
    # Export to PNG
    print("  Exporting to PNG via kaleido...")
    img_bytes = fig.to_image(format="png", width=1150, height=700, scale=2)
    print(f"  PNG size: {len(img_bytes)/1024:.0f} KB")
    
    # Save file copy
    png_path = Path("outputs/reports/before_21classes_scatter.png")
    png_path.write_bytes(img_bytes)
    print(f"  Saved: {png_path}")
    
    # Base64 encode
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    
    # Build HTML section
    before_section = f'''
<!-- BEFORE_IMAGE_SECTION_START -->
<div style="background:white; border-radius:10px; padding:20px 30px; margin-bottom:10px; box-shadow:0 2px 8px rgba(0,0,0,0.06);">
    <h3 style="color:#dc2626; margin:0 0 8px 0; font-size:14px;">BEFORE: 21 Classes (T7+T17 merged, T22 still present)</h3>
    <p style="color:#64748b; margin-bottom:12px; font-size:12px;">Static snapshot of the cluster map before T22 redistribution and Issue Type grouping. Compare with Section 8 at the bottom.</p>
    <div style="text-align:center;">
        <img src="data:image/png;base64,{b64}" 
             alt="Before: 21-class cluster map with T22"
             style="max-width:100%; height:auto; border:1px solid #e2e8f0; border-radius:6px;" />
    </div>
</div>
<!-- BEFORE_IMAGE_SECTION_END -->
'''
    
    # Insert into report - right before Section 1's div
    if "BEFORE_IMAGE_SECTION_START" in content:
        start = content.find("<!-- BEFORE_IMAGE_SECTION_START -->")
        end = content.find("<!-- BEFORE_IMAGE_SECTION_END -->") + len("<!-- BEFORE_IMAGE_SECTION_END -->")
        content = content[:start] + before_section + content[end:]
        print("  Replaced existing BEFORE image section.")
    else:
        # Find Section 1 heading and insert before its parent div
        section1_marker = '<!-- Section 1: Cluster Map -->'
        idx = content.find(section1_marker)
        if idx < 0:
            # Fallback: find first <div class="section">
            idx = content.find('<div class="section">')
        content = content[:idx] + before_section + "\n" + content[idx:]
        print("  Inserted BEFORE image above Section 1.")
    
    REPORT_PATH.write_text(content, encoding="utf-8")
    size_kb = REPORT_PATH.stat().st_size / 1024
    print(f"\n  Report size: {size_kb:.0f} KB")
    print("  Done!")


if __name__ == "__main__":
    main()
