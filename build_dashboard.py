"""
build_dashboard.py
==================
One-click interactive HTML dashboard generator.
Reads Phase 1-4 outputs and produces a single leadership_report.html
with interactive Plotly charts (heatmap, tables, bar charts).

Usage:
    python build_dashboard.py

Output:
    outputs/reports/leadership_report.html (self-contained, interactive)
"""
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ===========================================================================
# Dynamic Discovery
# ===========================================================================
_PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = _PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Auto-discover input files
def discover_file(name: str) -> Path:
    """Search outputs/ and outputs/reports/ for a file."""
    for directory in [REPORTS_DIR, OUTPUTS_DIR]:
        path = directory / name
        if path.exists():
            return path
    sys.exit(f"ERROR: Could not find '{name}' in {OUTPUTS_DIR} or {REPORTS_DIR}")

TAXONOMY_PATH = discover_file("final_taxonomy.csv")
SCORECARD_PATH = discover_file("taxonomy_scorecard.csv")
CROSS_TAB_PATH = discover_file("cross_tabulation.csv")
CORPUS_PATH = discover_file("corpus_clean.csv")

OUTPUT_HTML = REPORTS_DIR / "leadership_report.html"


# ===========================================================================
# Data Loading
# ===========================================================================
def load_data():
    """Load all required CSVs."""
    taxonomy = pd.read_csv(TAXONOMY_PATH, keep_default_na=False)
    scorecard = pd.read_csv(SCORECARD_PATH, keep_default_na=False)
    cross_tab = pd.read_csv(CROSS_TAB_PATH, index_col=0)
    corpus = pd.read_csv(CORPUS_PATH, dtype=str, keep_default_na=False)
    return taxonomy, scorecard, cross_tab, corpus


# ===========================================================================
# Chart Builders
# ===========================================================================
def build_heatmap(cross_tab: pd.DataFrame, taxonomy: pd.DataFrame) -> str:
    """Platform vs Topic heatmap (interactive Plotly)."""
    # Map topic IDs to business names
    id_to_name = dict(zip(taxonomy["topic_id"].astype(str), taxonomy["business_name"]))

    # Prepare data
    topic_ids = [str(idx) for idx in cross_tab.index]
    topic_labels = [f"T{tid}: {id_to_name.get(tid, tid)[:35]}" for tid in topic_ids]
    platforms = cross_tab.columns.tolist()
    values = cross_tab.values

    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=platforms,
        y=topic_labels,
        colorscale="Blues",
        hovertemplate="Platform: %{x}<br>Topic: %{y}<br>Tasks: %{z}<extra></extra>",
        colorbar=dict(title="Task Count"),
    ))

    fig.update_layout(
        title=dict(text="Operational Heatmap: Platform vs Issue Type", font=dict(size=18)),
        xaxis=dict(title="Platform", tickfont=dict(size=11)),
        yaxis=dict(title="", tickfont=dict(size=10), autorange="reversed"),
        height=max(500, len(topic_ids) * 28),
        margin=dict(l=280, r=40, t=60, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_pillar_bar(taxonomy: pd.DataFrame) -> str:
    """Business Pillar distribution bar chart."""
    pillar_counts = taxonomy.groupby("business_pillar")["doc_count"].sum().sort_values(ascending=True)

    fig = go.Figure(go.Bar(
        x=pillar_counts.values,
        y=pillar_counts.index,
        orientation="h",
        marker_color="#4a90d9",
        hovertemplate="%{y}: %{x} tasks<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Task Volume by Business Pillar", font=dict(size=18)),
        xaxis=dict(title="Number of Tasks"),
        yaxis=dict(title=""),
        height=350,
        margin=dict(l=200, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_confidence_pie(taxonomy: pd.DataFrame) -> str:
    """Confidence distribution pie chart."""
    conf_counts = taxonomy["confidence"].value_counts()

    colors = {"HIGH": "#52c41a", "OK": "#fadb14", "LOW": "#ff4d4f"}
    fig = go.Figure(go.Pie(
        labels=conf_counts.index.tolist(),
        values=conf_counts.values.tolist(),
        marker_colors=[colors.get(c, "#999") for c in conf_counts.index],
        hole=0.4,
        hovertemplate="%{label}: %{value} topics (%{percent})<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Topic Confidence Distribution", font=dict(size=18)),
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_coherence_bar(taxonomy: pd.DataFrame) -> str:
    """Per-topic coherence scores."""
    df = taxonomy.sort_values("coherence", ascending=True)
    colors = ["#ff4d4f" if c < 0.25 else "#fadb14" if c < 0.35 else "#52c41a" for c in df["coherence"]]

    fig = go.Figure(go.Bar(
        x=df["coherence"].values,
        y=[f"T{tid}: {name[:30]}" for tid, name in zip(df["topic_id"], df["business_name"])],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>Coherence: %{x:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Cluster Coherence Scores", font=dict(size=18)),
        xaxis=dict(title="Coherence (Avg Intra-Cluster Similarity)", range=[0, 1]),
        yaxis=dict(title="", tickfont=dict(size=10)),
        height=max(400, len(df) * 25),
        margin=dict(l=280, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_scorecard_table(scorecard: pd.DataFrame) -> str:
    """Scorecard as a Plotly table."""
    colors = []
    for s in scorecard["status"]:
        if s in ("PASS", "GOOD"):
            colors.append("#d4edda")
        elif s == "PENDING":
            colors.append("#fff3cd")
        else:
            colors.append("#f8d7da")

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Metric</b>", "<b>Value</b>", "<b>Status</b>"],
            fill_color="#2c3e50",
            font=dict(color="white", size=13),
            align="left",
        ),
        cells=dict(
            values=[scorecard["metric"], scorecard["value"], scorecard["status"]],
            fill_color=[["white"] * len(scorecard), ["white"] * len(scorecard), colors],
            font=dict(size=12),
            align="left",
            height=30,
        ),
    ))

    fig.update_layout(
        title=dict(text="Quality Scorecard", font=dict(size=18)),
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_taxonomy_table(taxonomy: pd.DataFrame) -> str:
    """Full taxonomy as interactive Plotly table."""
    display_cols = ["topic_id", "business_pillar", "business_name", "doc_count", "pct", "confidence", "action"]
    df = taxonomy[display_cols].copy()

    conf_colors = []
    for c in df["confidence"]:
        if c == "HIGH":
            conf_colors.append("#d4edda")
        elif c == "LOW":
            conf_colors.append("#f8d7da")
        else:
            conf_colors.append("#fff3cd")

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>ID</b>", "<b>Business Pillar</b>", "<b>Category Name</b>",
                    "<b>Tasks</b>", "<b>%</b>", "<b>Confidence</b>", "<b>Action</b>"],
            fill_color="#2c3e50",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=[df[c].tolist() for c in display_cols],
            fill_color=[["white"]*len(df), ["white"]*len(df), ["white"]*len(df),
                       ["white"]*len(df), ["white"]*len(df), conf_colors, ["white"]*len(df)],
            font=dict(size=11),
            align="left",
            height=28,
        ),
    ))

    fig.update_layout(
        title=dict(text="Complete Taxonomy Mapping", font=dict(size=18)),
        height=max(400, len(df) * 30 + 80),
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_cluster_scatter(taxonomy: pd.DataFrame, corpus: pd.DataFrame) -> str:
    """
    2D UMAP scatter plot using MERGED labels from training_data.csv.
    Each color = one final class (after merges applied).
    """
    from umap import UMAP as UMAPModel

    emb_path = Path("outputs/embeddings.npy")
    embeddings = np.load(str(emb_path))

    # Use merged labels from training_data.csv (reflects T7+T17 merge)
    train_path = Path("outputs/training_data.csv")
    if train_path.exists():
        train_df = pd.read_csv(train_path, dtype=str, keep_default_na=False)
        # Align by task_id
        label_map = dict(zip(train_df["task_id"], train_df["label"]))
        labels = [label_map.get(tid, "Unknown") for tid in corpus["task_id"]]
    else:
        # Fallback: use taxonomy topic_id mapping
        id_to_name = dict(zip(taxonomy["topic_id"].astype(str), taxonomy["business_name"]))
        labels = ["Unknown"] * len(corpus)

    # 2D UMAP for visualization
    print("    Computing 2D UMAP projection for scatter plot...")
    umap_2d = UMAPModel(n_neighbors=15, n_components=2, min_dist=0.1, metric="cosine", random_state=42)
    coords_2d = umap_2d.fit_transform(embeddings)

    # Distinct colors for up to 22 classes
    DISTINCT_COLORS = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
        "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
        "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#e6beff",
        "#ffe119", "#fabebe",
    ]

    unique_labels = sorted(set(labels))
    label_to_color = {l: DISTINCT_COLORS[i % len(DISTINCT_COLORS)] for i, l in enumerate(unique_labels)}

    # Compute centroids per label
    plot_df = pd.DataFrame({"x": coords_2d[:, 0], "y": coords_2d[:, 1], "label": labels})
    centroids = plot_df.groupby("label").agg({"x": "mean", "y": "mean"}).reset_index()
    centroids["count"] = [sum(1 for l in labels if l == lbl) for lbl in centroids["label"]]

    fig = go.Figure()

    # One trace per label
    for lbl in unique_labels:
        mask = plot_df["label"] == lbl
        subset = plot_df[mask]
        count = len(subset)
        short_name = lbl[:30]
        fig.add_trace(go.Scatter(
            x=subset["x"], y=subset["y"],
            mode="markers",
            marker=dict(size=5, color=label_to_color[lbl], opacity=0.55,
                        line=dict(width=0.3, color="white")),
            name=f"{short_name} ({count})",
            legendgroup=lbl,
            hovertemplate=f"<b>{short_name}</b><extra></extra>",
        ))

    # Centroid diamonds
    fig.add_trace(go.Scatter(
        x=centroids["x"], y=centroids["y"],
        mode="markers+text",
        marker=dict(
            size=16,
            color=[label_to_color[l] for l in centroids["label"]],
            symbol="diamond",
            line=dict(color="black", width=2),
        ),
        text=[f"{l[:18]}" for l in centroids["label"]],
        textposition="top center",
        textfont=dict(size=8, color="#333"),
        name="Centroids",
        showlegend=True,
        hovertemplate="<b>%{text}</b><br>Tasks: %{customdata}<extra>CENTROID</extra>",
        customdata=centroids["count"].tolist(),
    ))

    fig.update_layout(
        title=dict(text=f"Cluster Map ({len(unique_labels)} Classes, Merged)", font=dict(size=18)),
        xaxis=dict(title="UMAP-1", showgrid=True, gridcolor="#eee", zeroline=False),
        yaxis=dict(title="UMAP-2", showgrid=True, gridcolor="#eee", zeroline=False),
        height=800,
        width=1200,
        margin=dict(l=50, r=280, t=60, b=50),
        paper_bgcolor="white",
        plot_bgcolor="#fafcff",
        legend=dict(title="Classes (click to toggle)", font=dict(size=9),
                    yanchor="top", y=1, xanchor="left", x=1.02),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_similarity_heatmap(corpus: pd.DataFrame) -> str:
    """
    Class-vs-Class cosine similarity heatmap.
    Shows how similar each pair of classes is (based on mean embeddings).
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    emb_path = Path("outputs/embeddings.npy")
    embeddings = np.load(str(emb_path))

    train_path = Path("outputs/training_data.csv")
    if not train_path.exists():
        return "<p>training_data.csv not found - run apply_merges.py first</p>"

    train_df = pd.read_csv(train_path, dtype=str, keep_default_na=False)
    label_map = dict(zip(train_df["task_id"], train_df["label"]))
    labels = [label_map.get(tid, "Unknown") for tid in corpus["task_id"]]

    unique_labels = sorted(set(labels))

    # Compute mean embedding (centroid) per class
    centroids = []
    for lbl in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == lbl]
        centroid = embeddings[indices].mean(axis=0)
        centroids.append(centroid)

    centroids_matrix = np.array(centroids)
    sim_matrix = cos_sim(centroids_matrix)

    # Short labels for display
    short_labels = [l[:28] for l in unique_labels]

    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix,
        x=short_labels,
        y=short_labels,
        colorscale="RdYlGn",
        zmin=0.4,
        zmax=1.0,
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Similarity: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Cosine Sim"),
    ))

    fig.update_layout(
        title=dict(text="Class Similarity Matrix (Centroid Cosine Similarity)", font=dict(size=18)),
        height=700,
        width=900,
        margin=dict(l=200, r=40, t=60, b=200),
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
        paper_bgcolor="white",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ===========================================================================
# HTML Assembly
# ===========================================================================
def assemble_html(charts: dict, stats: dict, timestamp: str) -> str:
    """Combine all charts into a single self-contained HTML file."""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Operational Taxonomy Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f0f2f5;
    color: #1a1a2e;
    line-height: 1.6;
}}
.container {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 30px 40px;
}}
header {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: white;
    padding: 40px;
    text-align: center;
    margin-bottom: 30px;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}}
header h1 {{
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 8px;
}}
header p {{
    color: #a0aec0;
    font-size: 14px;
}}
.kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 30px;
}}
.kpi-card {{
    background: white;
    border-radius: 10px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.2s;
}}
.kpi-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
.kpi-value {{
    font-size: 36px;
    font-weight: 800;
    color: #2563eb;
    line-height: 1.2;
}}
.kpi-label {{
    font-size: 12px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 6px;
}}
.section {{
    background: white;
    border-radius: 10px;
    padding: 30px;
    margin-bottom: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}
.section h2 {{
    font-size: 20px;
    color: #1e293b;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid #e2e8f0;
}}
.grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
}}
@media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
.footer {{
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    margin-top: 40px;
    padding: 20px;
}}
</style>
</head>
<body>
<div class="container">

<header>
    <h1>Operational Taxonomy Dashboard</h1>
    <p>Task Classification Model | {stats['total_docs']:,} Asana Tasks | Generated: {timestamp}</p>
</header>

<!-- KPI Row -->
<div class="kpi-row">
    <div class="kpi-card"><div class="kpi-value">{stats['n_pillars']}</div><div class="kpi-label">Business Pillars</div></div>
    <div class="kpi-card"><div class="kpi-value">{stats['n_topics']}</div><div class="kpi-label">Topic Clusters</div></div>
    <div class="kpi-card"><div class="kpi-value">{stats['total_docs']:,}</div><div class="kpi-label">Tasks Classified</div></div>
    <div class="kpi-card"><div class="kpi-value">0%</div><div class="kpi-label">Outlier Rate</div></div>
    <div class="kpi-card"><div class="kpi-value">{stats['avg_coherence']:.2f}</div><div class="kpi-label">Avg Coherence</div></div>
    <div class="kpi-card"><div class="kpi-value">{stats['high_conf_pct']:.0f}%</div><div class="kpi-label">High Confidence</div></div>
</div>

<!-- Section 1: Cluster Map -->
<div class="section">
    <h2>1. Topic Cluster Map (Centroid Visualization)</h2>
    <p style="color:#64748b; margin-bottom:10px; font-size:13px;">Each dot is a task, colored by Business Pillar. Stars mark cluster centroids (size = cluster count). Hover for details. Zoom to explore.</p>
    {charts['cluster_scatter']}
</div>

<!-- Section 2: Pillar Distribution -->
<div class="section">
    <h2>2. Task Volume by Business Pillar</h2>
    {charts['pillar_bar']}
</div>

<!-- Section 3: Heatmap -->
<div class="section">
    <h2>3. Platform vs Issue Type Heatmap</h2>
    <p style="color:#64748b; margin-bottom:10px; font-size:13px;">Hover over cells to see exact task counts. Shows which platforms generate which operational issues.</p>
    {charts['heatmap']}
</div>

<!-- Section 3b: Similarity Heatmap -->
<div class="section">
    <h2>3b. Class Similarity Matrix</h2>
    <p style="color:#64748b; margin-bottom:10px; font-size:13px;">Shows how similar each class is to every other class (based on centroid cosine similarity). Green = highly similar (merge candidate). Red = very different (good separation).</p>
    {charts['similarity_heatmap']}
</div>

<!-- Section 4: Quality -->
<div class="grid-2">
    <div class="section">
        <h2>4. Quality Scorecard</h2>
        {charts['scorecard']}
    </div>
    <div class="section">
        <h2>5. Confidence Distribution</h2>
        {charts['confidence_pie']}
    </div>
</div>

<!-- Section 6: Coherence -->
<div class="section">
    <h2>6. Cluster Coherence Scores</h2>
    <p style="color:#64748b; margin-bottom:10px; font-size:13px;">Green = high coherence (tight cluster), Yellow = moderate, Red = low (consider splitting or reviewing).</p>
    {charts['coherence_bar']}
</div>

<!-- Section 7: Full Taxonomy -->
<div class="section">
    <h2>7. Complete Taxonomy Mapping</h2>
    {charts['taxonomy_table']}
</div>

<div class="footer">
    <p>Generated by <strong>build_dashboard.py</strong> | Re-run any time to update with latest results</p>
    <p>{timestamp}</p>
</div>

</div>
</body>
</html>"""
    return html


# ===========================================================================
# Main
# ===========================================================================
def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 60)
    print("BUILD DASHBOARD: Interactive Leadership Report")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    # Discover and load
    print(f"\n[1/4] Discovering files...")
    print(f"  Taxonomy:       {TAXONOMY_PATH}")
    print(f"  Scorecard:      {SCORECARD_PATH}")
    print(f"  Cross-tab:      {CROSS_TAB_PATH}")
    print(f"  Corpus:         {CORPUS_PATH}")

    print(f"\n[2/4] Loading data...")
    taxonomy, scorecard, cross_tab, corpus = load_data()
    total_docs = len(corpus)
    n_pillars = taxonomy["business_pillar"].nunique()
    n_topics = len(taxonomy)
    avg_coherence = taxonomy["coherence"].mean()
    high_conf = (taxonomy["confidence"] == "HIGH").sum()
    high_conf_pct = high_conf / n_topics * 100

    stats = {
        "total_docs": total_docs,
        "n_pillars": n_pillars,
        "n_topics": n_topics,
        "avg_coherence": avg_coherence,
        "high_conf_pct": high_conf_pct,
    }
    print(f"  {total_docs:,} tasks | {n_pillars} pillars | {n_topics} topics")

    # Build charts
    print(f"\n[3/4] Building interactive charts...")
    charts = {
        "cluster_scatter": build_cluster_scatter(taxonomy, corpus),
        "similarity_heatmap": build_similarity_heatmap(corpus),
        "heatmap": build_heatmap(cross_tab, taxonomy),
        "pillar_bar": build_pillar_bar(taxonomy),
        "confidence_pie": build_confidence_pie(taxonomy),
        "coherence_bar": build_coherence_bar(taxonomy),
        "scorecard": build_scorecard_table(scorecard),
        "taxonomy_table": build_taxonomy_table(taxonomy),
    }
    print(f"  Built 8 interactive Plotly charts")

    # Assemble HTML
    print(f"\n[4/4] Assembling dashboard...")
    html = assemble_html(charts, stats, timestamp)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    file_size = OUTPUT_HTML.stat().st_size / 1024
    print(f"\n  Saved: {OUTPUT_HTML}")
    print(f"  Size:  {file_size:.0f} KB")

    print(f"\n{'=' * 60}")
    print(f"DASHBOARD READY")
    print(f"{'=' * 60}")
    print(f"  Open in browser: {OUTPUT_HTML.resolve()}")
    print(f"  All charts are interactive (zoom, hover, pan)")
    print(f"  Re-run this script to update after any model changes")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
