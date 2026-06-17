"""
append_after_scatter.py
========================
Appends a NEW section to leadership_report.html showing the final
20-class centroid scatter plot colored by 8 Issue Types.

This is the "AFTER" view - to be compared with the existing Section 1
which shows the "BEFORE" state (21 classes including T22).

Does NOT modify any existing content in the report.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

REPORT_PATH = Path("outputs/reports/leadership_report.html")
TRAIN_PATH = Path("outputs/training_data.csv")
EMB_PATH = Path("outputs/embeddings.npy")
CORPUS_PATH = Path("outputs/corpus_clean.csv")

# 8 Issue Type colors (high contrast, colorblind-friendly)
PILLAR_COLORS = {
    "Portfolio & Investment Operations": "#e6194b",
    "Valuation & Pricing": "#3cb44b",
    "Account & Data Onboarding": "#4363d8",
    "Transaction & Cash Flow Processing": "#f58231",
    "Client Reporting & Deliverables": "#911eb4",
    "Ownership, Structure & Obligations": "#42d4f4",
    "Data Quality & Governance": "#f032e6",
    "Operational Administration": "#808000",
}


def build_after_scatter():
    """Build the 20-class scatter colored by 8 Issue Types."""
    from umap import UMAP as UMAPModel

    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH, dtype=str, keep_default_na=False)
    corpus = pd.read_csv(CORPUS_PATH, dtype=str, keep_default_na=False)
    embeddings = np.load(str(EMB_PATH))

    # Map task_id -> label and task_id -> pillar from training data
    label_map = dict(zip(train_df["task_id"], train_df["label"]))
    pillar_map = dict(zip(train_df["task_id"], train_df["business_pillar"]))

    labels = [label_map.get(tid, "Unknown") for tid in corpus["task_id"]]
    pillars = [pillar_map.get(tid, "Unknown") for tid in corpus["task_id"]]

    # Verify: should be 20 labels, 8 pillars
    unique_labels = sorted(set(labels) - {"Unknown"})
    unique_pillars = sorted(set(pillars) - {"Unknown"})
    print(f"  Labels: {len(unique_labels)} classes")
    print(f"  Pillars: {len(unique_pillars)} Issue Types")

    # 2D UMAP
    print("  Computing 2D UMAP projection...")
    umap_2d = UMAPModel(n_neighbors=15, n_components=2, min_dist=0.1,
                        metric="cosine", random_state=42)
    coords_2d = umap_2d.fit_transform(embeddings)

    # Build DataFrame
    plot_df = pd.DataFrame({
        "x": coords_2d[:, 0],
        "y": coords_2d[:, 1],
        "label": labels,
        "pillar": pillars,
    })

    # Remove unknowns (shouldn't be any, but just in case)
    plot_df = plot_df[plot_df["pillar"] != "Unknown"].copy()

    fig = go.Figure()

    # One trace per sub-type, colored by pillar
    for pillar in sorted(PILLAR_COLORS.keys()):
        pillar_df = plot_df[plot_df["pillar"] == pillar]
        sub_types = sorted(pillar_df["label"].unique())

        for st in sub_types:
            mask = pillar_df["label"] == st
            subset = pillar_df[mask]
            count = len(subset)
            short_name = st[:30]

            fig.add_trace(go.Scatter(
                x=subset["x"], y=subset["y"],
                mode="markers",
                marker=dict(size=5, color=PILLAR_COLORS[pillar], opacity=0.55,
                            line=dict(width=0.3, color="white")),
                name=f"{short_name} ({count})",
                legendgroup=pillar,
                legendgrouptitle_text=pillar[:35],
                hovertemplate=f"<b>{short_name}</b><br>Issue Type: {pillar}<extra></extra>",
            ))

    # Centroids per sub-type (colored by pillar)
    centroids_data = plot_df.groupby(["label", "pillar"]).agg(
        x=("x", "mean"), y=("y", "mean"), count=("x", "count")
    ).reset_index()

    fig.add_trace(go.Scatter(
        x=centroids_data["x"], y=centroids_data["y"],
        mode="markers+text",
        marker=dict(
            size=14,
            color=[PILLAR_COLORS.get(p, "#999") for p in centroids_data["pillar"]],
            symbol="diamond",
            line=dict(color="black", width=2),
        ),
        text=[f"{l[:18]}" for l in centroids_data["label"]],
        textposition="top center",
        textfont=dict(size=7, color="#333"),
        name="Centroids (20 Sub-types)",
        showlegend=True,
        hovertemplate="<b>%{text}</b><br>Tasks: %{customdata}<extra>CENTROID</extra>",
        customdata=centroids_data["count"].tolist(),
    ))

    # Also add large star markers for each Issue Type centroid
    pillar_centroids = plot_df.groupby("pillar").agg(
        x=("x", "mean"), y=("y", "mean"), count=("x", "count")
    ).reset_index()

    fig.add_trace(go.Scatter(
        x=pillar_centroids["x"], y=pillar_centroids["y"],
        mode="markers+text",
        marker=dict(
            size=22,
            color=[PILLAR_COLORS.get(p, "#999") for p in pillar_centroids["pillar"]],
            symbol="star",
            line=dict(color="black", width=2.5),
        ),
        text=[f"{p[:20]}" for p in pillar_centroids["pillar"]],
        textposition="bottom center",
        textfont=dict(size=9, color="#111", family="Arial Black"),
        name="Issue Type Centroids (8)",
        showlegend=True,
        hovertemplate="<b>%{text}</b><br>Tasks: %{customdata}<extra>ISSUE TYPE</extra>",
        customdata=pillar_centroids["count"].tolist(),
    ))

    fig.update_layout(
        title=dict(
            text=f"Final Cluster Map: {len(unique_labels)} Sub-types × {len(unique_pillars)} Issue Types (After T22 Redistribution)",
            font=dict(size=16)
        ),
        xaxis=dict(title="UMAP-1", showgrid=True, gridcolor="#eee", zeroline=False),
        yaxis=dict(title="UMAP-2", showgrid=True, gridcolor="#eee", zeroline=False),
        height=850,
        width=1250,
        margin=dict(l=50, r=320, t=70, b=50),
        paper_bgcolor="white",
        plot_bgcolor="#fafcff",
        legend=dict(
            title="Issue Types / Sub-types (click to toggle)",
            font=dict(size=9),
            yanchor="top", y=1, xanchor="left", x=1.02,
            groupclick="togglegroup",
        ),
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_issue_type_table():
    """Build the Issue Type -> Sub-type mapping table HTML."""
    train_df = pd.read_csv(TRAIN_PATH, dtype=str, keep_default_na=False)

    pillar_data = train_df.groupby("business_pillar").apply(
        lambda g: g.groupby("label").size().to_dict()
    ).to_dict()

    rows_html = ""
    for i, (pillar, subtypes) in enumerate(sorted(pillar_data.items()), 1):
        total = sum(subtypes.values())
        pct = total / len(train_df) * 100
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
            <th style="padding:10px;">Issue Type (Color)</th>
            <th style="padding:10px;">Sub-types (Count)</th>
            <th style="padding:10px;">Total</th>
            <th style="padding:10px;">%</th>
        </tr>
        {rows_html}
    </table>"""


def main():
    print("=" * 60)
    print("APPENDING AFTER-STATE CENTROID PLOT TO DASHBOARD")
    print("=" * 60)

    # Build the scatter
    scatter_html = build_after_scatter()
    table_html = build_issue_type_table()

    # Build the new section
    new_section = f"""
<!-- AFTER_SCATTER_SECTION_START -->
<div class="section" style="border-top: 4px solid #16a34a; margin-top:40px;">
    <h2>8. Final Cluster Map: After T22 Redistribution (20 Classes × 8 Issue Types)</h2>
    <p style="color:#64748b; margin-bottom:10px; font-size:13px;">
        <strong>Compare with Section 1 above.</strong> T22 (Asset Transfers &amp; Movements, 95 docs) has been
        redistributed via KNN (K=7) + 18 manual semantic overrides. Points are now colored by
        <strong>Issue Type</strong> (8 colors). Diamond markers = sub-type centroids. Star markers = Issue Type centroids.
        Click legend groups to toggle entire Issue Types.
    </p>
    <div style="background:#f0fdf4; border-left:4px solid #16a34a; padding:12px 16px; margin-bottom:16px; font-size:13px;">
        <strong>Method progression:</strong> Section 1 shows the raw BERTopic discovery (21 classes). 
        This section shows the refined taxonomy after: (1) T7+T17 merge, (2) T22 KNN redistribution, 
        (3) hierarchical grouping into 8 Issue Types.
    </div>
    {scatter_html}

    <h3 style="margin-top:30px; color:#1e293b;">Issue Type Grouping Summary</h3>
    {table_html}
</div>
<!-- AFTER_SCATTER_SECTION_END -->
"""

    # Read existing report and append before closing tags
    print("\nReading existing report...")
    content = open(REPORT_PATH, "r", encoding="utf-8").read()

    # Check if already appended
    if "AFTER_SCATTER_SECTION_START" in content:
        # Replace existing section
        start_marker = "<!-- AFTER_SCATTER_SECTION_START -->"
        end_marker = "<!-- AFTER_SCATTER_SECTION_END -->"
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker) + len(end_marker)
        content = content[:start_idx] + new_section + content[end_idx:]
        print("  Replaced existing AFTER scatter section.")
    else:
        # Insert before </div></body></html> at the end
        # Find the last </div> before </body>
        close_body = content.rfind("</body>")
        # Insert before the closing container div + body
        close_container = content.rfind("</div>", 0, close_body)
        content = content[:close_container] + new_section + "\n" + content[close_container:]
        print("  Appended new AFTER scatter section.")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    size_kb = REPORT_PATH.stat().st_size / 1024
    print(f"\n  Saved: {REPORT_PATH}")
    print(f"  Size: {size_kb:.0f} KB")
    print("\n" + "=" * 60)
    print("DONE - Open the report to see Section 1 (before) vs Section 8 (after)")
    print("=" * 60)


if __name__ == "__main__":
    main()
