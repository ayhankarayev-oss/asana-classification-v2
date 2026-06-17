"""
build_regrouped_dashboard.py
==============================
Creates a NEW dashboard page (leadership_report_v2.html) with regrouped Issue Types.
Does NOT modify the original leadership_report.html.

Changes from v1:
- MERGE sub-types: "Security & Position Updates" + "Position Cleanup & Deduplication" -> "Security & Position Management"
- REGROUP: "Cash Flow & Distribution Management" + "Commitment & Capital Call Tracking" under new "Capital & Cash Flow Management"
- MOVE: "Debt & Lending Instruments" -> "Portfolio & Investment Operations"
- RENAME: "Ownership, Structure & Obligations" -> "Ownership & Trust Structure" (standalone, 139 docs)
- Result: 8 Issue Types, 19 Sub-types (down from 20)
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

TRAIN_PATH = Path("outputs/training_data.csv")
CORPUS_PATH = Path("outputs/corpus_clean.csv")
EMB_PATH = Path("outputs/embeddings.npy")
OUTPUT_HTML = Path("outputs/reports/leadership_report_v2.html")

# Sub-type merge: two labels become one
SUBTYPE_MERGE = {
    "Security & Position Updates": "Security & Position Management",
    "Position Cleanup & Deduplication": "Security & Position Management",
}

# Issue Type reassignments
PILLAR_OVERRIDES = {
    "Cash Flow & Distribution Management": "Capital & Cash Flow Management",
    "Commitment & Capital Call Tracking": "Capital & Cash Flow Management",
    "Debt & Lending Instruments": "Portfolio & Investment Operations",
    "Ownership & Trust Structure": "Ownership & Trust Structure",
    "Security & Position Management": "Portfolio & Investment Operations",
}

# 8 Issue Type colors
PILLAR_COLORS = {
    "Portfolio & Investment Operations": "#e6194b",
    "Valuation & Pricing": "#3cb44b",
    "Account & Data Onboarding": "#4363d8",
    "Client Reporting & Deliverables": "#911eb4",
    "Capital & Cash Flow Management": "#f58231",
    "Ownership & Trust Structure": "#42d4f4",
    "Data Quality & Governance": "#f032e6",
    "Operational Administration": "#808000",
}


def apply_v2_transforms(train_df):
    """Apply sub-type merges and pillar reassignments."""
    df = train_df.copy()
    # Step 1: Merge sub-types
    df["label"] = df["label"].replace(SUBTYPE_MERGE)
    # Step 2: Reassign pillars
    df["business_pillar"] = df["label"].map(
        lambda lbl: PILLAR_OVERRIDES.get(lbl, df.loc[df["label"] == lbl, "business_pillar"].iloc[0] if len(df[df["label"] == lbl]) > 0 else "Unknown")
    )
    # Fix: apply pillar overrides properly
    for label, pillar in PILLAR_OVERRIDES.items():
        mask = df["label"] == label
        if mask.any():
            df.loc[mask, "business_pillar"] = pillar
    # Labels not in PILLAR_OVERRIDES keep their existing pillar
    return df


def build_regrouped_scatter(train_v2, corpus, embeddings):
    """Build the scatter plot with 8 Issue Type colors (regrouped)."""
    from umap import UMAP as UMAPModel

    label_map = dict(zip(train_v2["task_id"], train_v2["label"]))
    pillar_map = dict(zip(train_v2["task_id"], train_v2["business_pillar"]))

    labels = [label_map.get(tid, "Unknown") for tid in corpus["task_id"]]
    pillars = [pillar_map.get(tid, "Unknown") for tid in corpus["task_id"]]

    unique_labels = sorted(set(labels) - {"Unknown"})
    unique_pillars = sorted(set(pillars) - {"Unknown"})
    print(f"  Sub-types: {len(unique_labels)} | Issue Types: {len(unique_pillars)}")

    # 2D UMAP
    print("  Computing 2D UMAP...")
    umap_2d = UMAPModel(n_neighbors=15, n_components=2, min_dist=0.1,
                        metric="cosine", random_state=42)
    coords_2d = umap_2d.fit_transform(embeddings)

    plot_df = pd.DataFrame({
        "x": coords_2d[:, 0], "y": coords_2d[:, 1],
        "label": labels, "pillar": pillars,
    })
    plot_df = plot_df[plot_df["pillar"] != "Unknown"].copy()

    fig = go.Figure()

    for pillar in sorted(PILLAR_COLORS.keys()):
        pillar_df = plot_df[plot_df["pillar"] == pillar]
        sub_types = sorted(pillar_df["label"].unique())
        for st in sub_types:
            subset = pillar_df[pillar_df["label"] == st]
            count = len(subset)
            fig.add_trace(go.Scatter(
                x=subset["x"], y=subset["y"],
                mode="markers",
                marker=dict(size=5, color=PILLAR_COLORS[pillar], opacity=0.55,
                            line=dict(width=0.3, color="white")),
                name=f"{st[:30]} ({count})",
                legendgroup=pillar,
                legendgrouptitle_text=pillar[:35],
                hovertemplate=f"<b>{st[:30]}</b><br>Issue Type: {pillar}<extra></extra>",
            ))

    # Sub-type centroids (diamonds)
    centroids_data = plot_df.groupby(["label", "pillar"]).agg(
        x=("x", "mean"), y=("y", "mean"), count=("x", "count")
    ).reset_index()

    fig.add_trace(go.Scatter(
        x=centroids_data["x"], y=centroids_data["y"],
        mode="markers+text",
        marker=dict(size=14,
                    color=[PILLAR_COLORS.get(p, "#999") for p in centroids_data["pillar"]],
                    symbol="diamond", line=dict(color="black", width=2)),
        text=[f"{l[:18]}" for l in centroids_data["label"]],
        textposition="top center",
        textfont=dict(size=7, color="#333"),
        name=f"Sub-type Centroids ({len(centroids_data)})",
        showlegend=True,
        hovertemplate="<b>%{text}</b><br>Tasks: %{customdata}<extra></extra>",
        customdata=centroids_data["count"].tolist(),
    ))

    # Issue Type centroids (stars)
    pillar_centroids = plot_df.groupby("pillar").agg(
        x=("x", "mean"), y=("y", "mean"), count=("x", "count")
    ).reset_index()

    fig.add_trace(go.Scatter(
        x=pillar_centroids["x"], y=pillar_centroids["y"],
        mode="markers+text",
        marker=dict(size=22,
                    color=[PILLAR_COLORS.get(p, "#999") for p in pillar_centroids["pillar"]],
                    symbol="star", line=dict(color="black", width=2.5)),
        text=[f"{p[:22]}" for p in pillar_centroids["pillar"]],
        textposition="bottom center",
        textfont=dict(size=8, color="#111", family="Arial Black"),
        name="Issue Type Centroids (8)",
        showlegend=True,
        hovertemplate="<b>%{text}</b><br>Tasks: %{customdata}<extra></extra>",
        customdata=pillar_centroids["count"].tolist(),
    ))

    fig.update_layout(
        title=dict(text=f"Regrouped: {len(unique_labels)} Sub-types × {len(unique_pillars)} Issue Types (v2)",
                   font=dict(size=16)),
        xaxis=dict(title="UMAP-1", showgrid=True, gridcolor="#eee", zeroline=False),
        yaxis=dict(title="UMAP-2", showgrid=True, gridcolor="#eee", zeroline=False),
        height=850, width=1250,
        margin=dict(l=50, r=320, t=70, b=50),
        paper_bgcolor="white", plot_bgcolor="#fafcff",
        legend=dict(title="Issue Types / Sub-types", font=dict(size=9),
                    yanchor="top", y=1, xanchor="left", x=1.02, groupclick="togglegroup"),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_grouping_table(train_v2):
    """Build the regrouped Issue Type table."""
    rows_html = ""
    pillar_data = {}
    for pillar in sorted(train_v2["business_pillar"].unique()):
        sub = train_v2[train_v2["business_pillar"] == pillar]
        pillar_data[pillar] = sub.groupby("label").size().to_dict()

    for i, (pillar, subtypes) in enumerate(sorted(pillar_data.items()), 1):
        total = sum(subtypes.values())
        pct = total / len(train_v2) * 100
        subtypes_str = ", ".join(f"{name} ({count})" for name, count in sorted(subtypes.items(), key=lambda x: -x[1]))
        bg = "background:#eff6ff;" if i % 2 == 0 else ""
        color = PILLAR_COLORS.get(pillar, "#999")
        rows_html += f"""
        <tr style="{bg}">
            <td style="padding:8px; font-weight:bold;">{i}</td>
            <td style="padding:8px; font-weight:bold;"><span style="display:inline-block;width:12px;height:12px;background:{color};border-radius:2px;margin-right:6px;vertical-align:middle;"></span>{pillar}</td>
            <td style="padding:8px; font-size:12px;">{subtypes_str}</td>
            <td style="padding:8px; font-weight:bold;">{total}</td>
            <td style="padding:8px;">{pct:.1f}%</td>
        </tr>"""

    return f"""
    <table style="width:100%; border-collapse:collapse; font-size:13px; margin:15px 0;">
        <tr style="background:#1e293b; color:white;">
            <th style="padding:10px;">#</th>
            <th style="padding:10px;">Issue Type</th>
            <th style="padding:10px;">Sub-types (Count)</th>
            <th style="padding:10px;">Total</th>
            <th style="padding:10px;">%</th>
        </tr>
        {rows_html}
        <tr style="background:#1e293b; color:white; font-weight:bold;">
            <td style="padding:10px;" colspan="3">TOTAL</td>
            <td style="padding:10px;">{len(train_v2):,}</td>
            <td style="padding:10px;">100%</td>
        </tr>
    </table>"""


def build_changes_html():
    """Visual diff of v1 -> v2."""
    return """
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin:20px 0;">
        <div style="background:#fef2f2; border:1px solid #fca5a5; border-radius:8px; padding:16px;">
            <h4 style="margin:0 0 10px 0; color:#dc2626;">Changes from v1</h4>
            <ul style="font-size:13px; margin:0; padding-left:18px; line-height:1.8;">
                <li><strong>MERGED sub-types:</strong> "Security & Position Updates" + "Position Cleanup & Deduplication" &rarr; <strong>"Security & Position Management"</strong> (172 docs)</li>
                <li><strong>DISSOLVED:</strong> "Transaction & Cash Flow Processing" (1 sub-type) and "Ownership, Structure & Obligations" (3 sub-types) — reorganized</li>
                <li><strong>MOVED:</strong> "Debt & Lending Instruments" &rarr; "Portfolio & Investment Operations"</li>
            </ul>
        </div>
        <div style="background:#f0fdf4; border:1px solid #86efac; border-radius:8px; padding:16px;">
            <h4 style="margin:0 0 10px 0; color:#16a34a;">New in v2</h4>
            <ul style="font-size:13px; margin:0; padding-left:18px; line-height:1.8;">
                <li><strong>NEW Issue Type:</strong> "Capital & Cash Flow Management" (Cash Flow + Commitment = 221 docs)</li>
                <li><strong>STANDALONE:</strong> "Ownership & Trust Structure" (139 docs) — its own Issue Type now</li>
                <li><strong>EXPANDED:</strong> "Portfolio & Investment Operations" now includes Debt & merged Security/Position (560 docs)</li>
            </ul>
        </div>
    </div>
    <div style="background:#eff6ff; border-left:4px solid #2563eb; padding:12px 16px; border-radius:4px; margin-top:12px; font-size:13px;">
        <strong>Rationale:</strong>
        (1) Security updates and position cleanup are the same operational action — modifying how an instrument is represented.
        (2) Capital calls and cash distributions are two sides of the same fund cash lifecycle.
        (3) Debt/lending are financial instruments managed as portfolio positions.
        (4) Ownership/trust is a distinct legal-structural domain separate from cash flows.
    </div>
    """


def build_comparison_table():
    """Side-by-side v1 vs v2 comparison."""
    return """
    <table style="width:100%; border-collapse:collapse; font-size:12px; margin:15px 0;">
        <tr style="background:#1e293b; color:white;">
            <th style="padding:8px;">v1 Issue Type</th>
            <th style="padding:8px;">v1 Sub-types</th>
            <th style="padding:8px;">v2 Issue Type</th>
            <th style="padding:8px;">v2 Sub-types</th>
        </tr>
        <tr style="background:#f8fafc;">
            <td style="padding:6px;">Portfolio & Investment Operations</td>
            <td style="padding:6px;">New Private Investment Setup, Security & Position Updates, Position Cleanup & Dedup</td>
            <td style="padding:6px;"><strong>Portfolio & Investment Operations</strong></td>
            <td style="padding:6px;">New Private Investment Setup, <strong>Security & Position Management</strong> (merged), <strong>Debt & Lending Instruments</strong> (moved in)</td>
        </tr>
        <tr>
            <td style="padding:6px;">Transaction & Cash Flow Processing</td>
            <td style="padding:6px;">Cash Flow & Distribution Management</td>
            <td style="padding:6px; color:#dc2626;"><s>Dissolved</s></td>
            <td style="padding:6px;">&rarr; moved to Capital & Cash Flow Management</td>
        </tr>
        <tr style="background:#f8fafc;">
            <td style="padding:6px;">Ownership, Structure & Obligations</td>
            <td style="padding:6px;">Ownership & Trust Structure, Commitment & Capital Call, Debt & Lending</td>
            <td style="padding:6px; color:#dc2626;"><s>Dissolved</s></td>
            <td style="padding:6px;">&rarr; split into 3 different Issue Types</td>
        </tr>
        <tr>
            <td style="padding:6px; color:#16a34a;">&mdash;</td>
            <td style="padding:6px;">&mdash;</td>
            <td style="padding:6px; color:#16a34a;"><strong>Capital & Cash Flow Management (NEW)</strong></td>
            <td style="padding:6px;">Cash Flow & Distribution Mgmt, Commitment & Capital Call Tracking</td>
        </tr>
        <tr style="background:#f8fafc;">
            <td style="padding:6px; color:#16a34a;">&mdash;</td>
            <td style="padding:6px;">&mdash;</td>
            <td style="padding:6px; color:#16a34a;"><strong>Ownership & Trust Structure (NEW)</strong></td>
            <td style="padding:6px;">Ownership & Trust Structure (standalone)</td>
        </tr>
    </table>
    """


def build_similarity_heatmap(train_v2, corpus, embeddings):
    """Build class-vs-class cosine similarity heatmap for v2 labels."""
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    label_map = dict(zip(train_v2["task_id"], train_v2["label"]))
    labels = [label_map.get(tid, "Unknown") for tid in corpus["task_id"]]
    unique_labels = sorted(set(labels) - {"Unknown"})

    # Compute centroids per sub-type
    centroids = []
    for lbl in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == lbl]
        centroid = embeddings[indices].mean(axis=0)
        centroids.append(centroid)

    centroids_matrix = np.array(centroids)
    sim_matrix = cos_sim(centroids_matrix)

    # Get pillar for coloring
    pillar_map = dict(zip(train_v2["task_id"], train_v2["business_pillar"]))
    label_to_pillar = {}
    for lbl in unique_labels:
        mask = train_v2["label"] == lbl
        if mask.any():
            label_to_pillar[lbl] = train_v2.loc[mask, "business_pillar"].iloc[0]

    short_labels = [f"{l[:25]}" for l in unique_labels]

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
        title=dict(text="Sub-type Similarity Matrix (v2, 19 Classes)", font=dict(size=16)),
        height=650,
        width=850,
        margin=dict(l=180, r=40, t=60, b=180),
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
        paper_bgcolor="white",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_summary_html():
    """Build a methodology summary section."""
    return """
    <div style="font-size:13px; line-height:1.8;">
        <h3 style="margin:0 0 12px 0; color:#1e293b;">Project Timeline & Methods</h3>
        <table style="width:100%; border-collapse:collapse; font-size:12px;">
            <tr style="background:#1e293b; color:white;">
                <th style="padding:8px;">Phase</th>
                <th style="padding:8px;">Method</th>
                <th style="padding:8px;">Result</th>
            </tr>
            <tr style="background:#f8fafc;">
                <td style="padding:6px; font-weight:bold;">1. Preprocessing</td>
                <td style="padding:6px;">Custom PII masking + Microsoft Presidio (ML) + entity extraction</td>
                <td style="padding:6px;">2,683 clean, anonymized task texts</td>
            </tr>
            <tr>
                <td style="padding:6px; font-weight:bold;">2. Embedding</td>
                <td style="padding:6px;">all-mpnet-base-v2 (768-dim sentence transformer)</td>
                <td style="padding:6px;">Dense vector representation for each task</td>
            </tr>
            <tr style="background:#f8fafc;">
                <td style="padding:6px; font-weight:bold;">3. Clustering</td>
                <td style="padding:6px;">BERTopic (UMAP + HDBSCAN, min_cluster=30, soft-clustering)</td>
                <td style="padding:6px;">23 raw topics, 0% outliers</td>
            </tr>
            <tr>
                <td style="padding:6px; font-weight:bold;">4. Audit & Naming</td>
                <td style="padding:6px;">30-50 sample reviews per cluster, business-outcome naming</td>
                <td style="padding:6px;">All clusters renamed to reflect business actions</td>
            </tr>
            <tr style="background:#f8fafc;">
                <td style="padding:6px; font-weight:bold;">5. Merge (T7+T17)</td>
                <td style="padding:6px;">Cosine similarity (0.82) + sample validation</td>
                <td style="padding:6px;">23 → 22 topics</td>
            </tr>
            <tr>
                <td style="padding:6px; font-weight:bold;">6. T22 Redistribution</td>
                <td style="padding:6px;">KNN (K=7) + 18 manual semantic overrides</td>
                <td style="padding:6px;">22 → 21 → 20 sub-types (v1)</td>
            </tr>
            <tr style="background:#f8fafc;">
                <td style="padding:6px; font-weight:bold;">7. Hierarchical Grouping</td>
                <td style="padding:6px;">Ward's linkage dendrogram + semantic validation</td>
                <td style="padding:6px;">20 sub-types → 8 Issue Types (v1)</td>
            </tr>
            <tr>
                <td style="padding:6px; font-weight:bold;">8. Split Validation</td>
                <td style="padding:6px;">KMeans(k=2) + silhouette scoring per cluster</td>
                <td style="padding:6px;">2 candidates flagged (not split)</td>
            </tr>
            <tr style="background:#eff6ff; font-weight:bold;">
                <td style="padding:6px;">9. Regrouping (v2)</td>
                <td style="padding:6px;">Sub-type merge + Issue Type reorganization (semantic review)</td>
                <td style="padding:6px;">19 sub-types × 8 Issue Types (v2)</td>
            </tr>
        </table>
    </div>
    """


def main():
    print("=" * 60)
    print("BUILD REGROUPED DASHBOARD (v2) — 8 Issue Types × 19 Sub-types")
    print("=" * 60)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load and transform
    print("\n[1/4] Loading and transforming data...")
    train_df = pd.read_csv(TRAIN_PATH, dtype=str, keep_default_na=False)
    corpus = pd.read_csv(CORPUS_PATH, dtype=str, keep_default_na=False)
    embeddings = np.load(str(EMB_PATH))

    train_v2 = apply_v2_transforms(train_df)
    n_labels = train_v2["label"].nunique()
    n_pillars = train_v2["business_pillar"].nunique()
    print(f"  v2: {n_labels} sub-types, {n_pillars} Issue Types, {len(train_v2)} docs")

    # Build charts
    print("\n[2/4] Building scatter plot...")
    scatter_html = build_regrouped_scatter(train_v2, corpus, embeddings)
    print("\n[3/4] Building similarity matrix...")
    similarity_html = build_similarity_heatmap(train_v2, corpus, embeddings)
    table_html = build_grouping_table(train_v2)
    changes_html = build_changes_html()
    comparison_html = build_comparison_table()
    summary_html = build_summary_html()

    # Assemble HTML
    print("\n[4/4] Assembling page...")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Regrouped Taxonomy v2 — 8 Issue Types × 19 Sub-types</title>
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
    background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
    color: white;
    padding: 40px;
    text-align: center;
    margin-bottom: 30px;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}}
header h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
header p {{ color: #a0aec0; font-size: 14px; }}
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
.kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
    margin-bottom: 30px;
}}
.kpi-card {{
    background: white;
    border-radius: 10px;
    padding: 20px 16px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}
.kpi-value {{ font-size: 32px; font-weight: 800; color: #2563eb; line-height: 1.2; }}
.kpi-label {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }}
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
    <h1>Regrouped Taxonomy v2: 8 Issue Types × 19 Sub-types</h1>
    <p>Refined grouping based on semantic review | Generated: {timestamp}</p>
</header>

<div class="kpi-row">
    <div class="kpi-card"><div class="kpi-value">8</div><div class="kpi-label">Issue Types</div></div>
    <div class="kpi-card"><div class="kpi-value">19</div><div class="kpi-label">Sub-types</div></div>
    <div class="kpi-card"><div class="kpi-value">2,683</div><div class="kpi-label">Tasks</div></div>
    <div class="kpi-card"><div class="kpi-value">1</div><div class="kpi-label">Sub-type Merged</div></div>
    <div class="kpi-card"><div class="kpi-value">2</div><div class="kpi-label">Issue Types Dissolved</div></div>
    <div class="kpi-card"><div class="kpi-value">2</div><div class="kpi-label">Issue Types Created</div></div>
</div>

<!-- Changes -->
<div class="section">
    <h2>What Changed (v1 → v2)</h2>
    {changes_html}
</div>

<!-- Comparison Table -->
<div class="section">
    <h2>v1 → v2 Comparison</h2>
    <p style="color:#64748b; margin-bottom:12px; font-size:13px;">Only rows that changed are shown. Unchanged Issue Types (Account & Data Onboarding, Client Reporting, Data Quality, Operational Admin, Valuation & Pricing) remain identical.</p>
    {comparison_html}
</div>

<!-- New Grouping Table -->
<div class="section">
    <h2>Final v2 Issue Type Mapping</h2>
    {table_html}
</div>

<!-- Scatter Plot -->
<div class="section">
    <h2>Cluster Map: 19 Sub-types × 8 Issue Types (v2)</h2>
    <p style="color:#64748b; margin-bottom:10px; font-size:13px;">
        Same UMAP projection. Colors reflect the 8 regrouped Issue Types.
        Diamond = sub-type centroid. Star = Issue Type centroid. Click legend groups to toggle.
    </p>
    {scatter_html}
</div>

<!-- Similarity Matrix -->
<div class="section">
    <h2>Sub-type Similarity Matrix (v2)</h2>
    <p style="color:#64748b; margin-bottom:10px; font-size:13px;">
        Cosine similarity between class centroids after merging. Green = high similarity (potential further merge). Red = distinct classes (good separation).
        Compare with v1 similarity matrix in the original report.
    </p>
    {similarity_html}
</div>

<!-- Summary -->
<div class="section">
    <h2>Methodology Summary (Full Pipeline)</h2>
    <p style="color:#64748b; margin-bottom:12px; font-size:13px;">
        Complete sequence of methods used from raw data to final v2 taxonomy.
    </p>
    {summary_html}
</div>

<div class="footer">
    <p>Generated by <strong>build_regrouped_dashboard.py</strong> | {timestamp}</p>
    <p><a href="leadership_report.html">&larr; Back to original report (v1: 8 Issue Types × 20 Sub-types)</a></p>
</div>

</div>
</body>
</html>"""

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    size_kb = OUTPUT_HTML.stat().st_size / 1024
    print(f"\n  Saved: {OUTPUT_HTML}")
    print(f"  Size: {size_kb:.0f} KB")
    print(f"\n  Original (v1): leadership_report.html — UNCHANGED")
    print(f"  Regrouped (v2): leadership_report_v2.html — NEW")
    print("=" * 60)


if __name__ == "__main__":
    main()
