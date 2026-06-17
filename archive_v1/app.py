"""
TOS Asana Task Classification Dashboard
========================================
Run:  streamlit run app.py
URL:  http://localhost:8501
"""

import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TOS Asana Classification Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
[data-testid="metric-container"] {
    background:#f4f6fb; border:1px solid #dde3ef;
    border-radius:10px; padding:14px 18px;
}
h2 { border-bottom:2px solid #e3e7ed; padding-bottom:5px; margin-top:1rem; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df  = pd.read_csv("complaints_classified.csv", dtype=str, keep_default_na=False)
    exp = pd.read_csv("topic_explanations.csv",    dtype=str, keep_default_na=False)

    df["topic_id"]    = pd.to_numeric(df["topic_id"],    errors="coerce").fillna(-1).astype(int)
    exp["topic_id"]   = pd.to_numeric(exp["topic_id"],   errors="coerce").fillna(-1).astype(int)
    exp["topic_size"] = pd.to_numeric(exp["topic_size"], errors="coerce").fillna(0).astype(int)
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce").fillna(0.0)
    df["dup_count"] = pd.to_numeric(df.get("dup_count", pd.Series([1]*len(df))), errors="coerce").fillna(1).astype(int)

    for c in [f"ctfidf_{i:02d}" for i in range(1, 11)]:
        exp[c] = pd.to_numeric(exp.get(c, pd.Series([0.0]*len(exp))), errors="coerce").fillna(0.0)

    # Attempt to parse dates
    if "Created At" in df.columns:
        df["created_dt"] = pd.to_datetime(df["Created At"], errors="coerce")
    else:
        df["created_dt"] = pd.NaT

    return df, exp

df, exp = load_data()

# ── Constants ─────────────────────────────────────────────────────────────────
TOTAL        = len(df)
N_CLASSIFIED = int((df["topic_id"] != -1).sum())
N_OUTLIERS   = int((df["topic_id"] == -1).sum())
N_TOPICS     = int((exp["topic_id"] != -1).sum())
# Original row count before deduplication (sum of dup_count restores the full corpus)
N_ORIGINAL   = int(df["dup_count"].sum())
N_DEDUPED    = N_ORIGINAL - TOTAL   # tasks removed by dedup

n_tagged_before = int((df.get("Issue type", pd.Series([""])).str.strip() != "").sum())
pct_before   = round(n_tagged_before / TOTAL * 100, 1)
pct_after    = round(N_CLASSIFIED    / TOTAL * 100, 1)
pct_outlier  = round(N_OUTLIERS      / TOTAL * 100, 1)

# Business label per topic_id
blabel_map = (
    df[df["topic_id"] != -1]
    .groupby("topic_id")["business_label"]
    .first()
    .to_dict()
)

# Cluster summary table
clusters = (
    exp[exp["topic_id"] != -1].copy()
    .sort_values("topic_size", ascending=False)
    .reset_index(drop=True)
)
clusters["pct"]            = (clusters["topic_size"] / TOTAL * 100).round(1)
clusters["business_label"] = clusters["topic_id"].map(blabel_map).fillna(clusters["topic_label"])
clusters["ctfidf_01_f"]    = pd.to_numeric(clusters["ctfidf_01"], errors="coerce").fillna(0.0)

def short(s, n=36):
    return s[:n] + "…" if len(s) > n else s

clusters["label_short"] = clusters["business_label"].apply(lambda x: short(x, 36))

PALETTE = (
    px.colors.qualitative.Plotly
    + px.colors.qualitative.Pastel
    + px.colors.qualitative.Safe
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 TOS Asana Task Classification Dashboard")
st.caption(
    f"BERTopic v0.17 · ZeroShot · all-MiniLM-L6-v2 · {TOTAL:,} tasks · "
    f"**Open tabs below to explore each view**"
)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Unique Tasks",         f"{TOTAL:,}",        f"{N_ORIGINAL:,} before dedup (−{N_DEDUPED:,})")
k2.metric("Topics Discovered",    f"{N_TOPICS}")
k3.metric("Tagged Before",        f"{pct_before}%",    f"{n_tagged_before:,} tasks")
k4.metric("Classified After",     f"{pct_after}%",     f"+{pct_after - pct_before:.1f}pp")
k5.metric("Unclassified",         f"{pct_outlier}%",   f"{N_OUTLIERS} tasks", delta_color="inverse")
k6.metric("Coverage Gain",        f"+{pct_after - pct_before:.1f}pp", "manually tagged → model")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🔄 Before vs After",
    "📊 Cluster Sizes",
    "🔍 Topic Explorer",
    "🏷️ Classify New Task",
    "📈 Confidence & Quality",
    "🗓️ Time Trends",
    "ℹ️ Model Details",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — BEFORE vs AFTER
# ═════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Before vs After: Coverage and Classification Quality")

    st.info(
        f"**The Problem:** Only **{pct_before}%** of {N_ORIGINAL:,} Asana tasks had a manual "
        f"'Issue type' tag. The remaining **{100-pct_before:.1f}%** ({N_ORIGINAL-n_tagged_before:,} tasks) "
        f"had no classification at all — making it impossible to analyse workload distribution or spot trends.\n\n"
        f"**The Solution:** After deduplication ({N_ORIGINAL:,} → {TOTAL:,} unique tasks, removing "
        f"{N_DEDUPED:,} identical recurring tasks), BERTopic automatically classified **{pct_after}%** "
        f"into **{N_TOPICS} coherent business categories**. Only **{pct_outlier}%** remain unclassified."
    )

    col_a, col_b = st.columns(2)

    # Before donut
    with col_a:
        st.markdown("#### Before — Asana Manual Tags")
        if "Issue type" in df.columns:
            tag_vc = df["Issue type"].str.strip().value_counts()
            tag_df = tag_vc[tag_vc.index != ""].reset_index()
            tag_df.columns = ["label", "count"]
            untagged = pd.DataFrame([{"label": "⬜ Untagged", "count": TOTAL - n_tagged_before}])
            tag_df   = pd.concat([untagged, tag_df], ignore_index=True)
            tag_df["pct"] = (tag_df["count"] / TOTAL * 100).round(1)

            fig = px.pie(
                tag_df, values="count", names="label",
                title=f"{TOTAL:,} tasks — {len(tag_df)-1} Asana labels + untagged",
                hole=0.5,
                color_discrete_sequence=["#c8c8c8"] + px.colors.qualitative.Set2,
            )
            fig.update_traces(
                textinfo="percent+label", textposition="inside",
                textfont_size=10, pull=[0.06]+[0]*(len(tag_df)-1),
            )
            fig.update_layout(height=420, showlegend=False, margin=dict(t=40,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("'Issue type' column not found.")

    # After donut
    with col_b:
        st.markdown("#### After — BERTopic Topic Clusters")
        after_df = clusters[["business_label","topic_size","pct"]].copy()
        after_df = pd.concat([
            after_df,
            pd.DataFrame([{"business_label":"⬜ Unclassified","topic_size":N_OUTLIERS,"pct":pct_outlier}])
        ], ignore_index=True)

        fig = px.pie(
            after_df, values="topic_size", names="business_label",
            title=f"{TOTAL:,} tasks — {N_TOPICS} BERTopic clusters",
            hole=0.5,
            color_discrete_sequence=PALETTE + ["#c8c8c8"],
        )
        fig.update_traces(textinfo="percent", textposition="inside", textfont_size=10)
        fig.update_layout(
            height=420,
            legend=dict(orientation="v", x=1.01, font_size=9),
            margin=dict(t=40,b=10,l=10,r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Coverage stacked bar
    st.markdown("#### Coverage Improvement at a Glance")
    fig = go.Figure()
    fig.add_bar(
        name="Classified", x=["Before (manual)", "After (BERTopic)"],
        y=[pct_before, pct_after],
        marker_color=["#90caf9","#1565c0"],
        text=[f"{pct_before}%", f"{pct_after}%"],
        textposition="inside", textfont=dict(color="white", size=16),
    )
    fig.add_bar(
        name="Unclassified", x=["Before (manual)", "After (BERTopic)"],
        y=[100-pct_before, pct_outlier],
        marker_color=["#e0e0e0","#eeeeee"],
        text=[f"{100-pct_before:.1f}%", f"{pct_outlier}%"],
        textposition="inside", textfont=dict(color="#555", size=14),
    )
    fig.update_layout(
        barmode="stack", height=200,
        yaxis=dict(title="% of tasks", range=[0,108]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=10,b=30,l=50,r=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cross-tab
    if "Issue type" in df.columns:
        st.markdown("#### Where did each Asana label land in BERTopic?")
        st.caption("Each bar = one existing Asana Issue type label. Colours = which BERTopic cluster absorbed those tasks.")
        tagged_df = df[df["Issue type"].str.strip() != ""].copy()
        top_asana = tagged_df["Issue type"].value_counts().head(10).index.tolist()
        ct = (
            tagged_df[tagged_df["Issue type"].isin(top_asana)]
            .groupby(["Issue type","business_label"]).size().reset_index(name="count")
        )
        fig = px.bar(
            ct, x="count", y="Issue type", color="business_label", orientation="h",
            labels={"count":"Tasks","Issue type":"Asana label","business_label":"BERTopic cluster"},
            color_discrete_sequence=PALETTE,
        )
        fig.update_layout(
            height=420, legend=dict(orientation="v", x=1.01, font_size=9),
            margin=dict(t=20,b=20,l=180,r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLUSTER SIZES
# ═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("All 15 Topic Clusters — Volume and Keyword Profiles")

    # Horizontal bar
    fig = px.bar(
        clusters.sort_values("topic_size"),
        x="pct", y="label_short", orientation="h",
        color="pct", color_continuous_scale="Blues",
        text=clusters.sort_values("topic_size").apply(
            lambda r: f"{r['topic_size']:,}  ({r['pct']}%)", axis=1),
        title="Cluster sizes as % of total corpus",
        labels={"pct":"% of corpus","label_short":""},
    )
    fig.update_traces(textposition="outside", textfont_size=11)
    fig.update_layout(
        height=600, coloraxis_showscale=False,
        margin=dict(t=40,b=30,l=10,r=120),
        yaxis=dict(tickfont_size=12),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Treemap — shows relative sizes very intuitively
    st.markdown("#### Treemap — relative cluster sizes")
    fig = px.treemap(
        clusters, path=["business_label"], values="topic_size",
        color="pct", color_continuous_scale="Blues",
        title="Area proportional to task count",
        hover_data={"topic_size":True,"pct":True,"kw_01":True},
    )
    fig.update_traces(textinfo="label+percent root", textfont_size=12)
    fig.update_layout(height=480, margin=dict(t=40,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)

    # Coherence chart
    st.markdown("#### Cluster Coherence — how tightly defined is each topic?")
    st.caption(
        "c-TF-IDF score of the top keyword: higher = more unique vocabulary = tighter cluster. "
        "Dark blue = strong; light grey = diffuse catch-all."
    )
    coh = clusters.sort_values("ctfidf_01_f", ascending=False).copy()
    coh["signal"] = coh["ctfidf_01_f"].apply(
        lambda s: "Strong (≥0.10)" if s>=0.10
        else ("Good (0.05–0.10)" if s>=0.05
        else ("Moderate (0.02–0.05)" if s>=0.02 else "Weak (<0.02)"))
    )
    fig = px.bar(
        coh, x="label_short", y="ctfidf_01_f",
        color="signal",
        color_discrete_map={
            "Strong (≥0.10)":"#1565c0","Good (0.05–0.10)":"#42a5f5",
            "Moderate (0.02–0.05)":"#90caf9","Weak (<0.02)":"#d0d0d0",
        },
        category_orders={"signal":["Strong (≥0.10)","Good (0.05–0.10)","Moderate (0.02–0.05)","Weak (<0.02)"]},
        text=coh["ctfidf_01_f"].apply(lambda x: f"{x:.3f}"),
        title="Top c-TF-IDF score per cluster (coherence proxy)",
        labels={"ctfidf_01_f":"Top c-TF-IDF","label_short":""},
    )
    fig.update_traces(textposition="outside", textfont_size=10)
    fig.update_layout(
        height=420,
        xaxis=dict(tickangle=-38, tickfont_size=10),
        margin=dict(t=50,b=130,l=50,r=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Full table
    st.markdown("#### Full Data Table")
    kw_cols = [f"kw_{i:02d}" for i in range(1, 6)]
    available = ["topic_id","business_label","topic_size","pct"] + [c for c in kw_cols if c in clusters.columns]
    tbl = clusters[available].copy()
    tbl.columns = ["ID","Business Label","Tasks","% Total"] + [f"KW {i}" for i in range(1, len(available)-3)]
    st.dataframe(tbl, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — TOPIC EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Topic Explorer — Drill Into Any Cluster")

    sel_label = st.selectbox(
        "Pick a topic cluster:", clusters["business_label"].tolist(), index=0
    )
    sel_row  = clusters[clusters["business_label"] == sel_label].iloc[0]
    sel_tid  = int(sel_row["topic_id"])
    sel_size = int(sel_row["topic_size"])
    sel_pct  = float(sel_row["pct"])
    rank_pos = int(clusters[clusters["business_label"] == sel_label].index[0]) + 1

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tasks in cluster", f"{sel_size:,}")
    m2.metric("% of corpus",       f"{sel_pct:.1f}%")
    m3.metric("Volume rank",       f"#{rank_pos} of {N_TOPICS}")
    top_s = float(sel_row["ctfidf_01_f"])
    m4.metric("Coherence (top c-TF-IDF)", f"{top_s:.4f}",
              "Strong" if top_s>=0.10 else ("Good" if top_s>=0.05 else "Moderate"))

    col_kw, col_tasks = st.columns([1, 1])

    with col_kw:
        st.markdown("**Top 10 Keywords (c-TF-IDF)**")
        kw_data = []
        for i in range(1, 11):
            kw    = str(sel_row.get(f"kw_{i:02d}", "")).strip()
            score = float(sel_row.get(f"ctfidf_{i:02d}", 0.0))
            if kw:
                kw_data.append({"Keyword": kw, "Score": score})
        if kw_data:
            kw_df = pd.DataFrame(kw_data).sort_values("Score")
            fig = px.bar(
                kw_df, x="Score", y="Keyword", orientation="h",
                color="Score", color_continuous_scale="Blues",
                text=kw_df["Score"].apply(lambda x: f"{x:.4f}"),
            )
            fig.update_traces(textposition="outside", textfont_size=11)
            fig.update_layout(
                height=340, coloraxis_showscale=False,
                margin=dict(t=10,b=20,l=10,r=80),
            )
            st.plotly_chart(fig, use_container_width=True)

            if top_s >= 0.10:   note = "🔵 **Strong** — very tight, well-defined cluster"
            elif top_s >= 0.05: note = "🟦 **Good** — clearly characterised"
            elif top_s >= 0.02: note = "🟧 **Moderate** — some overlap with adjacent topics"
            else:               note = "⬜ **Weak** — diffuse catch-all; mixed content expected"
            st.caption(f"Coherence: {note}")

    with col_tasks:
        st.markdown("**15 Sample Tasks from This Cluster**")
        sample_cols = ["Name","form_intent","confidence_score","dup_count"]
        available_cols = [c for c in sample_cols if c in df.columns]
        sdf = (
            df[df["topic_id"] == sel_tid][available_cols]
            .sample(min(15, sel_size), random_state=42)
            .reset_index(drop=True)
        )
        col_rename = {"Name":"Task Name","form_intent":"Form Intent",
                      "confidence_score":"Confidence","dup_count":"Repeats"}
        sdf = sdf.rename(columns=col_rename)
        sdf["Task Name"]   = sdf["Task Name"].str[:80]
        if "Form Intent" in sdf.columns:
            sdf["Form Intent"] = sdf["Form Intent"].str[:35]
        st.dataframe(sdf, use_container_width=True, hide_index=True, height=340)

    # Context bar
    st.markdown("**How this cluster compares in volume**")
    ctx = clusters.sort_values("pct", ascending=False).copy()
    ctx["highlight"] = ctx["business_label"].apply(
        lambda x: "Selected" if x == sel_label else "Other"
    )
    fig = px.bar(
        ctx, x="label_short", y="pct",
        color="highlight",
        color_discrete_map={"Selected":"#1565c0","Other":"#c8d8f0"},
        title=f"'{short(sel_label,28)}' — #{rank_pos} by volume ({sel_pct:.1f}%)",
        labels={"pct":"% of corpus","label_short":""},
        text=ctx["pct"].apply(lambda x: f"{x:.1f}%"),
    )
    fig.update_traces(textposition="outside", textfont_size=9)
    fig.update_layout(
        height=320, showlegend=False,
        xaxis=dict(tickangle=-38, tickfont_size=9),
        margin=dict(t=50,b=130,l=50,r=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Keyword detail table
    st.markdown("**Full Keyword Table**")
    if kw_data:
        full_kw = []
        for i in range(1, 11):
            kw    = str(sel_row.get(f"kw_{i:02d}","")).strip()
            score = float(sel_row.get(f"ctfidf_{i:02d}",0.0))
            if kw:
                sig = ("Very High" if score>=0.10 else ("High" if score>=0.05
                       else ("Moderate" if score>=0.02 else "Low")))
                full_kw.append({"Rank":i,"Keyword":kw,"c-TF-IDF Score":score,"Distinctiveness":sig})
        st.dataframe(pd.DataFrame(full_kw), use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — CLASSIFY NEW TASK
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    import re as _re
    import numpy as _np

    st.subheader("Classify a New Task")
    st.caption(
        "Paste a task name and/or notes — get an instant label, see which words drove "
        "the decision, and compare the top alternative categories."
    )

    @st.cache_resource
    def load_classifier():
        import joblib
        p = Path("issue_classifier.pkl")
        return joblib.load(p) if p.exists() else None

    # Lightweight platform → placeholder masking so the model sees the same
    # vocabulary it was trained on (preprocess_v2.py anonymised all platform names).
    _MASKS = [
        (r"\bAddepar(?:\s*-\s*(?:EU|Own))?\b", "[PORTFOLIO_PLATFORM]"),
        (r"\bArch\b",                            "[PORTFOLIO_PLATFORM]"),
        (r"\bVenn\b|\bNINES\b",                  "[PORTFOLIO_PLATFORM]"),
        (r"\bEgnyte\b",                          "[DOCUMENT_PLATFORM]"),
        (r"\bSharePoint\b",                      "[DOCUMENT_PLATFORM]"),
        (r"\bSalesforce\b|\bSFDC\b",             "[CRM_PLATFORM]"),
        (r"\bKnowLedger\b|\bQuickBooks?\b",      "[ACCOUNTING_PLATFORM]"),
        (r"\bEpicc\b",                           "[CLIENT_PORTAL]"),
        (r"\bFidelity(?:\s+Investments)?\b|Charles\s+Schwab\b|\bSchwab\b"
         r"|Morgan\s+Stanley\b|Merrill(?:\s+Lynch)?\b|Goldman(?:\s+Sachs)?\b"
         r"|\bDTCC?\b",                          "[CUSTODIAN]"),
        (r"\bJP\s*Morgan(?:\s+Chase)?\b|\bJPM\b|\bWells\s+Fargo\b|\bCiti(?:bank)?\b",
                                                 "[BANK]"),
        (r"\bMosaic(?:\s+Advisors)?\b|Elevate(?:\s+Advisory)?\b|\bTOS\b|\bSAOS\b",
                                                 "[ADVISOR_ORG]"),
        (r"\bPlaid\b|\bByAll\b|\bOrca\b",        "[DATA_AGGREGATOR]"),
        (r"\bAsana\b",                           "[PM_PLATFORM]"),
    ]

    def _quick_mask(text):
        for pat, repl in _MASKS:
            text = _re.sub(pat, repl, text, flags=_re.IGNORECASE)
        return text

    clf = load_classifier()

    if clf is None:
        st.error("Classifier not found. Run `train_classifier.py` first.")
    else:
        task_input = st.text_area(
            "Task description — paste the task name and/or notes:",
            height=130,
            placeholder=(
                "e.g. Please add the new private investment for Smith family as outlined "
                "below. Fund name: Blackstone BREIT. Asset class: Real Estate."
            ),
        )

        col_btn, col_note = st.columns([1, 5])
        clicked = col_btn.button("🔍  Classify", type="primary")
        col_note.caption(
            "Platform names (Addepar, Schwab, Egnyte …) are auto-normalised "
            "before classification to match training vocabulary."
        )

        if clicked and task_input.strip():
            text_masked = _quick_mask(task_input.strip())

            label   = clf.predict([text_masked])[0]
            proba   = clf.predict_proba([text_masked])[0]
            classes = list(clf.classes_)
            conf    = float(proba.max())

            # ── Coloured label badge ─────────────────────────────────────────
            if conf >= 0.70:
                bc, bt = "#1565c0", "High confidence"
            elif conf >= 0.50:
                bc, bt = "#f9a825", "Moderate confidence"
            else:
                bc, bt = "#e53935", "Low confidence — consider manual review"

            st.markdown(f"""
<div style="background:{bc}18; border-left:5px solid {bc}; border-radius:6px;
            padding:14px 18px; margin-bottom:8px;">
  <span style="font-size:1.35rem; font-weight:700; color:{bc};">{label}</span>
  <span style="margin-left:18px; font-size:0.9rem; color:#555;">
    {bt} &nbsp;·&nbsp; {conf:.1%}
  </span>
</div>""", unsafe_allow_html=True)
            st.progress(conf)
            st.markdown("---")

            col_l, col_r = st.columns([1.1, 1])

            # ── LEFT: words from YOUR input that drove the decision ──────────
            with col_l:
                st.markdown("##### Why this classification?")
                st.caption(
                    "Words / phrases from **your input** that pushed the model toward "
                    "this category.  Score = TF-IDF weight × logistic regression coefficient."
                )
                tfidf      = clf.named_steps["tfidf"]
                lr         = clf.named_steps["clf"]
                feat_names = tfidf.get_feature_names_out()
                X_vec      = tfidf.transform([text_masked])
                cidx       = classes.index(label)
                contribs   = X_vec.toarray()[0] * lr.coef_[cidx]

                top_idx = contribs.argsort()[::-1][:10]
                top_kw  = [(feat_names[i], float(contribs[i]))
                           for i in top_idx if contribs[i] > 0]

                if top_kw:
                    kw_df = pd.DataFrame(top_kw, columns=["Word / Phrase", "Score"])
                    fig = px.bar(
                        kw_df, x="Score", y="Word / Phrase", orientation="h",
                        color="Score", color_continuous_scale="Blues",
                        text=kw_df["Score"].apply(lambda v: f"{v:.4f}"),
                    )
                    fig.update_traces(textposition="outside", textfont_size=11)
                    fig.update_layout(
                        height=340, coloraxis_showscale=False,
                        margin=dict(t=10, b=10, l=10, r=75),
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(
                        "No strong keyword matches found. The input may use vocabulary "
                        "not seen in training, or the task description is very short."
                    )

            # ── RIGHT: cluster profile + runner-ups ──────────────────────────
            with col_r:
                st.markdown("##### What defines this category?")
                st.caption(
                    "Top c-TF-IDF keywords that **distinguish this cluster** across the "
                    "full training corpus (not your specific input)."
                )
                topic_rows = exp[exp["business_label"] == label]
                if not topic_rows.empty:
                    row = topic_rows.iloc[0]
                    ck = [(str(row.get(f"kw_{i:02d}","")), float(row.get(f"ctfidf_{i:02d}", 0)))
                          for i in range(1, 11)
                          if str(row.get(f"kw_{i:02d}","")).strip()]
                    if ck:
                        ck_df = pd.DataFrame(ck, columns=["Cluster Keyword", "c-TF-IDF"])
                        fig2 = px.bar(
                            ck_df, x="c-TF-IDF", y="Cluster Keyword", orientation="h",
                            color="c-TF-IDF", color_continuous_scale="Greens",
                            text=ck_df["c-TF-IDF"].apply(lambda v: f"{v:.4f}"),
                        )
                        fig2.update_traces(textposition="outside", textfont_size=10)
                        fig2.update_layout(
                            height=340, coloraxis_showscale=False,
                            margin=dict(t=10, b=10, l=10, r=75),
                            yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                st.markdown("##### Runner-up classifications")
                st.caption("How confident the model was about each alternative.")
                top4 = proba.argsort()[::-1][:4]
                alt_rows = []
                for rank, idx in enumerate(top4, 1):
                    filled = int(proba[idx] * 16)
                    bar = "█" * filled + "░" * (16 - filled)
                    alt_rows.append({
                        "#": rank,
                        "Category": classes[idx],
                        "Prob": f"{proba[idx]:.1%}",
                        "": bar,
                    })
                st.dataframe(pd.DataFrame(alt_rows), use_container_width=True, hide_index=True)

        elif clicked:
            st.warning("Please enter a task description before classifying.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — CONFIDENCE & QUALITY
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Classifier Confidence & Data Quality")

    classified_df = df[df["topic_id"] != -1].copy()

    # ── Confidence distribution overall ──────────────────────────────────────
    st.markdown("#### Overall Confidence Score Distribution")
    st.caption(
        "Confidence = probability assigned by the production classifier (issue_classifier.pkl). "
        "High-confidence tasks can be auto-routed; low-confidence tasks should go to a human reviewer."
    )
    fig = px.histogram(
        classified_df, x="confidence_score", nbins=20,
        color_discrete_sequence=["#1565c0"],
        title="Distribution of confidence scores across all classified tasks",
        labels={"confidence_score":"Confidence score","count":"Number of tasks"},
    )
    fig.add_vline(x=0.60, line_dash="dash", line_color="orange",
                  annotation_text="0.60 threshold (recommended)", annotation_position="top right")
    fig.update_layout(height=300, margin=dict(t=50,b=40,l=50,r=10))
    st.plotly_chart(fig, use_container_width=True)

    # Confidence buckets
    conf = classified_df["confidence_score"]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("High confidence (≥0.90)",   f"{int((conf>=0.90).sum()):,}",
              f"{(conf>=0.90).mean()*100:.1f}% of classified")
    c2.metric("Good (0.70–0.90)",           f"{int(((conf>=0.70)&(conf<0.90)).sum()):,}",
              f"{((conf>=0.70)&(conf<0.90)).mean()*100:.1f}%")
    c3.metric("Uncertain (0.50–0.70)",      f"{int(((conf>=0.50)&(conf<0.70)).sum()):,}",
              f"{((conf>=0.50)&(conf<0.70)).mean()*100:.1f}%")
    c4.metric("Review needed (<0.50)",      f"{int((conf<0.50).sum()):,}",
              f"{(conf<0.50).mean()*100:.1f}% — flag for manual review", delta_color="inverse")

    # ── Confidence per topic ──────────────────────────────────────────────────
    st.markdown("#### Average Confidence per Topic")
    st.caption("Topics where the classifier is least certain — these may need label refinement.")
    conf_by_topic = (
        classified_df.groupby("business_label")["confidence_score"]
        .agg(["mean","median","count"])
        .reset_index()
    )
    conf_by_topic.columns = ["Business Label","Mean Conf","Median Conf","Tasks"]
    conf_by_topic = conf_by_topic.sort_values("Mean Conf")

    fig = px.bar(
        conf_by_topic, x="Mean Conf", y="Business Label", orientation="h",
        color="Mean Conf", color_continuous_scale="RdYlGn",
        text=conf_by_topic["Mean Conf"].apply(lambda x: f"{x:.2f}"),
        title="Mean confidence score per cluster (lower = more uncertain assignments)",
    )
    fig.update_traces(textposition="outside", textfont_size=10)
    fig.add_vline(x=0.60, line_dash="dash", line_color="orange")
    fig.update_layout(
        height=560, coloraxis_showscale=False,
        margin=dict(t=50,b=30,l=10,r=80),
        xaxis=dict(range=[0,1.15]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Data quality: form_intent coverage ───────────────────────────────────
    st.markdown("#### Data Quality — How Much Signal Does Each Task Have?")
    st.caption(
        "Tasks with a 'form_intent' (from the Asana form 'How may we assist you?') have the "
        "richest classification signal. Name-only tasks are harder to classify reliably."
    )
    df["has_intent"] = df["form_intent"].str.strip().apply(lambda x: "Has form intent" if x else "Name-only")
    df["has_notes"]  = df["notes_clean"].str.strip().apply(lambda x: "Has notes" if x else "No notes")

    q1, q2 = st.columns(2)

    with q1:
        qi_vc = df["has_intent"].value_counts().reset_index()
        qi_vc.columns = ["Type","Count"]
        qi_vc["pct"]  = (qi_vc["Count"]/TOTAL*100).round(1)
        fig = px.pie(
            qi_vc, values="Count", names="Type",
            title="Form intent availability",
            color_discrete_sequence=["#1565c0","#c8d8f0"],
            hole=0.5,
        )
        fig.update_traces(textinfo="percent+label", textposition="inside", textfont_size=12)
        fig.update_layout(height=280, showlegend=False, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with q2:
        qn_vc = df["has_notes"].value_counts().reset_index()
        qn_vc.columns = ["Type","Count"]
        fig = px.pie(
            qn_vc, values="Count", names="Type",
            title="Notes field availability",
            color_discrete_sequence=["#42a5f5","#e0e0e0"],
            hole=0.5,
        )
        fig.update_traces(textinfo="percent+label", textposition="inside", textfont_size=12)
        fig.update_layout(height=280, showlegend=False, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Outlier sample ────────────────────────────────────────────────────────
    st.markdown("#### Unclassified Tasks (Outliers)")
    st.caption(
        f"These {N_OUTLIERS} tasks ({pct_outlier}%) scored below the 0.30 cosine-similarity threshold against "
        "all 15 zero-shot seeds and didn't form a dense HDBSCAN cluster (min_cluster_size=40). "
        "Review them to check for hidden patterns or lower min_cluster_size to recover more."
    )
    outlier_df = df[df["topic_id"] == -1][["Name","form_intent","Section/Column"]].copy()
    outlier_df.columns = ["Task Name","Form Intent","Status"]
    outlier_df["Task Name"] = outlier_df["Task Name"].str[:90]
    st.dataframe(outlier_df.reset_index(drop=True), use_container_width=True, hide_index=True, height=320)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — TIME TRENDS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Task Volume Over Time by Topic")

    has_dates = df["created_dt"].notna().sum() > 10

    if not has_dates:
        st.warning("No parseable 'Created At' dates found in the data. This tab requires date information.")
    else:
        classified_t = df[df["topic_id"] != -1].copy()
        classified_t = classified_t[classified_t["created_dt"].notna()].copy()
        classified_t["month"] = classified_t["created_dt"].dt.to_period("M").astype(str)

        # Overall monthly volume
        st.markdown("#### Total Task Volume by Month")
        monthly = classified_t.groupby("month").size().reset_index(name="count")
        fig = px.bar(
            monthly, x="month", y="count",
            color_discrete_sequence=["#1565c0"],
            title="Monthly task volume (all classified topics)",
            labels={"month":"Month","count":"Tasks"},
        )
        fig.update_layout(height=280, margin=dict(t=40,b=60,l=50,r=10),
                          xaxis=dict(tickangle=-45,tickfont_size=10))
        st.plotly_chart(fig, use_container_width=True)

        # Top-8 topics volume over time
        st.markdown("#### Monthly Volume — Top 8 Topics")
        top8 = clusters.head(8)["business_label"].tolist()
        t8_df = classified_t[classified_t["business_label"].isin(top8)].copy()
        t8_monthly = t8_df.groupby(["month","business_label"]).size().reset_index(name="count")

        fig = px.line(
            t8_monthly, x="month", y="count", color="business_label",
            title="Monthly task volume by topic (top 8 by size)",
            labels={"month":"Month","count":"Tasks","business_label":"Topic"},
            color_discrete_sequence=PALETTE,
            markers=True,
        )
        fig.update_layout(
            height=420,
            legend=dict(orientation="v", x=1.01, font_size=9),
            xaxis=dict(tickangle=-45, tickfont_size=10),
            margin=dict(t=50,b=60,l=50,r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Latest month breakdown
        latest_month = monthly["month"].max()
        st.markdown(f"#### Latest Month Breakdown — {latest_month}")
        latest_df = (
            classified_t[classified_t["month"] == latest_month]
            .groupby("business_label").size().reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        latest_df["label_short"] = latest_df["business_label"].apply(lambda x: short(x,38))
        fig = px.bar(
            latest_df, x="count", y="label_short", orientation="h",
            color="count", color_continuous_scale="Blues",
            text="count", title=f"Task count per topic in {latest_month}",
            labels={"count":"Tasks","label_short":""},
        )
        fig.update_traces(textposition="outside", textfont_size=11)
        fig.update_layout(
            height=480, coloraxis_showscale=False,
            margin=dict(t=40,b=20,l=10,r=60),
        )
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — MODEL DETAILS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("Model Architecture & Pipeline Details")

    st.markdown("""
#### How the Classification Pipeline Works

```
Client_support.csv (raw Asana export)
        │
        ▼  preprocess_v2.py
corpus_clean.csv (4,278 → 2,678 unique tasks)
  • Drops sub-tasks, KaizenBot alerts, Working Sessions
  • Extracts 'How may we assist you?' form field
  • Three-layer anonymisation:
      Layer 1 — structural regex: emails, URLs, phone numbers, entity IDs
      Layer 2 — named-entity blocklist: 10 typed platform categories
                 (e.g. Addepar → [PORTFOLIO_PLATFORM],
                       Fidelity → [CUSTODIAN], Egnyte → [DOCUMENT_PLATFORM])
      Layer 3 — spaCy NER (en_core_web_md): residual PERSON/ORG → [CLIENT]
  • Deduplication: 1,562 identical recurring tasks collapsed to
    representative entries (dup_count column records original frequency)
  • complaint_text = form_intent_masked + Name + notes_masked
        │
        ▼  model_pipeline.py  (BERTopic v0.17 · ZeroShot)
        │
        ├─ Step 1: Sentence-Transformer embedding
        │   Model : all-MiniLM-L6-v2  (384-dim vectors)
        │   Speed : ~1 min on CPU for 2,678 tasks
        │
        ├─ Step 2: UMAP dimensionality reduction
        │   384-dim → 5-dim  (preserves local neighbourhood structure)
        │   n_neighbors=20 · min_dist=0.0 · metric=cosine
        │
        ├─ Step 3: ZeroShot Pass 1 — seed matching
        │   15 hand-crafted seed descriptions (platform-agnostic vocabulary)
        │   Documents with cosine similarity ≥ 0.30 to any seed
        │   are assigned to that seed's topic
        │
        ├─ Step 4: HDBSCAN clustering (sklearn, no compilation needed)
        │   Remaining unmatched documents clustered automatically
        │   min_cluster_size=40 · metric=euclidean · eom selection
        │
        ├─ Step 5: c-TF-IDF keyword extraction
        │   Class-based TF-IDF — finds terms uniquely associated
        │   with each cluster vs the entire corpus
        │   Extended stopwords: anonymisation tokens + domain action verbs
        │
        └─ Step 6: MMR keyword reranking (diversity=0.3)
            Balances relevance vs diversity in keyword lists
            Avoids: [account, accounts, accounting, acct, ...]
        │
        ▼  Result: 15 topics · 15.8% outliers
complaints_classified.csv  +  topic_explanations.csv
        │
        ▼  train_classifier.py  (Phase 9)
issue_classifier.pkl
  • TF-IDF (ngram 1–2, 25k features) + Logistic Regression
  • Trained on BERTopic labels as ground truth
  • Accuracy: 63.0% test / 58.4% (5-fold CV)
  • Use for real-time scoring of new incoming tasks
```
""")

    # Seed topics used
    st.markdown("#### The 15 ZeroShot Seeds Used")
    st.caption("Seeds use platform-agnostic language — platform names (Addepar, Arch, Egnyte) are masked before embedding, so seeds must describe operations rather than tools.")
    seeds = [
        ("New Account & Data Feed Setup",            "New bank/brokerage/custodian account setup — connection, feed link, online portal access"),
        ("Portfolio Platform Account Updates",        "Account update, maintenance, close, rename, attribute change — classification, portfolio platform"),
        ("New Private Investment Entry",              "New private investment fund — create entry/position in portfolio platform, setup structure, direct owner"),
        ("Private Investment Updates & Valuations",   "Investment update — valuation, price change, commitment amount, capital, unfunded transaction"),
        ("Capital Call Audit & Monthly Statement Review", "Capital call, distribution — custodian payment processing, transfer, wire, verify, confirm receipt"),
        ("Capital Call Audit (Recurring Weekly)",     "Weekly periodic check all pending capital calls — verify status, mark complete, audit portfolio platform"),
        ("Cost Basis & Data Quality Fixes",           "Missing data attribute, cost basis quality — incorrect error, fix, export, reconcile"),
        ("Document Upload & Client Billing",          "Statement document upload — file retrieve/download/archive, K1, invoice, send billing, document platform"),
        ("Reporting & Performance Analytics",         "Report PDF — reporting, portfolio platform, view dashboard, performance, deliver, generate, quarterly"),
        ("Ownership Structure & Legal Entity Changes","Ownership structure — entity, LLC, trust, beneficiary, direct owner, setup change, legal"),
        ("Real Asset Transaction Audit",              "Recurring weekly/monthly transaction reconciliation — audit, verify, confirm status, accounting platform, operational"),
        ("New Account Setup / Onboarding",            "Onboarding new household — banker introduction, portal access, setup, client portal"),
        ("Follow-up & Pending Escalations",           "Follow up, reach out — email response pending, waiting, reminder, investigate, confirm, escalation"),
        ("Portfolio View & Access Configuration",     "Portfolio platform view/viewset — create, configure, user access, permissions, data aggregator, document"),
        ("General & Ad Hoc Requests",                 "General update, ad hoc, miscellaneous operational priority task — specify"),
    ]
    for i, (label, desc) in enumerate(seeds, 1):
        st.markdown(f"**{i}. {label}** — {desc}")

    # Performance table
    st.markdown("#### Production Classifier Performance (Phase 9)")
    try:
        report_path = Path("classifier_report.txt")
        if report_path.exists():
            with open(report_path, encoding="utf-8-sig") as f:
                report_txt = f.read()
            # Extract accuracy line
            for line in report_txt.splitlines():
                if "accuracy" in line.lower() and "%" in line:
                    st.success(line.strip())
                    break
            with st.expander("Full classifier report"):
                st.code(report_txt, language="text")
    except Exception:
        st.info("Run train_classifier.py to generate classifier_report.txt")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "**Files:** `complaints_classified.csv` · `topic_explanations.csv` · "
    "`cluster_summary_report.md` · `validation_report.txt` · "
    "`issue_classifier.pkl` · `topic_map.html` · `topic_barchart.html` · "
    "`topic_hierarchy.html` · `topic_heatmap.html`"
)
