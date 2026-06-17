"""
app.py
======
Streamlit app for Asana Task Classification.
- Displays the full interactive dashboard (leadership_report.html)
- Provides a prediction interface below for real-time classification
"""
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Asana Task Classification — TOS/Elevate Advisory",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Paths ---
REPORT_PATH = Path(__file__).parent / "outputs" / "reports" / "leadership_report.html"


# --- Cache the classifier so it only loads once ---
@st.cache_resource
def load_classifier():
    from predict import TaskClassifier
    return TaskClassifier()


# --- Main App ---
def main():
    st.title("Asana Task Classification System")
    st.caption("Family Office Data Management — TOS/Elevate Advisory")

    # Tab layout
    tab1, tab2 = st.tabs(["📊 Dashboard", "🔮 Predict"])

    # --- TAB 1: Dashboard ---
    with tab1:
        if REPORT_PATH.exists():
            html_content = REPORT_PATH.read_text(encoding="utf-8")
            st.components.v1.html(html_content, height=4000, scrolling=True)
        else:
            st.error(f"Dashboard not found at: {REPORT_PATH}")

    # --- TAB 2: Prediction Interface ---
    with tab2:
        st.header("Real-Time Task Classification")
        st.markdown(
            "Enter a task description below to predict its **Sub-type** (19 classes) "
            "and **Issue Type** (8 pillars) using the trained NLP model."
        )

        # Load classifier
        clf = load_classifier()

        # Input area
        text_input = st.text_area(
            "Task Description",
            height=120,
            placeholder="e.g., Please set up new custodian account for client and connect the bank feed to Addepar...",
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            predict_btn = st.button("Classify", type="primary", use_container_width=True)

        if predict_btn and text_input.strip():
            result = clf.predict(text_input)

            st.divider()

            # Main prediction
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Sub-type", result["sub_type"])
            with col_b:
                st.metric("Issue Type", result["issue_type"])
            with col_c:
                st.metric("Confidence", f"{result['confidence']:.1%}")

            # Alternatives
            if result["alternatives"]:
                st.markdown("**Alternative predictions:**")
                for i, alt in enumerate(result["alternatives"], 2):
                    st.markdown(
                        f"#{i}: **{alt['sub_type']}** → {alt['issue_type']} "
                        f"({alt['confidence']:.1%})"
                    )

        elif predict_btn:
            st.warning("Please enter a task description first.")

        # --- Batch mode ---
        st.divider()
        st.subheader("Batch Classification")
        st.markdown("Paste multiple task descriptions (one per line) for batch classification.")

        batch_input = st.text_area(
            "Batch Input (one task per line)",
            height=150,
            placeholder="Task 1 description\nTask 2 description\nTask 3 description",
            key="batch_input",
        )

        if st.button("Classify Batch", use_container_width=False):
            lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
            if lines:
                results = clf.predict_batch(lines)
                import pandas as pd

                df = pd.DataFrame([
                    {
                        "Task": text[:80] + "..." if len(text) > 80 else text,
                        "Sub-type": r["sub_type"],
                        "Issue Type": r["issue_type"],
                        "Confidence": f"{r['confidence']:.1%}",
                    }
                    for text, r in zip(lines, results)
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("Please enter at least one task description.")


if __name__ == "__main__":
    main()
