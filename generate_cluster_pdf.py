"""
generate_cluster_pdf.py
========================
Generates a PDF report analyzing all 19 clusters with:
- LLM-generated 6-7 sentence summary per cluster
- Analytics section (action types, platform mix, complexity, metrics)
- Full listing of all tasks with Asana GID headers
"""
import json
import pandas as pd
from pathlib import Path
from fpdf import FPDF

TRAIN_PATH = Path("outputs/training_data_v2.csv")
SUMMARIES_PATH = Path("outputs/cluster_summaries.json")
OUTPUT_PATH = Path("outputs/cluster_analysis_report.pdf")

# Issue Type ordering for logical grouping
PILLAR_ORDER = [
    "Account & Data Onboarding",
    "Portfolio & Investment Operations",
    "Capital & Cash Flow Management",
    "Client Reporting & Deliverables",
    "Data Quality & Governance",
    "Valuation & Pricing",
    "Operational Administration",
    "Ownership & Trust Structure",
]

# Action type keyword categories
ACTION_KEYWORDS = {
    "New/Setup": ["new account", "set up", "setup", "onboard", "add account", "create account",
                  "open account", "establish", "new client", "new fund", "new investment"],
    "Connect/Feed": ["feed", "connect", "data feed", "link", "integrate", "import", "automated",
                     "api", "portal access", "credentials"],
    "Fix/Resolve": ["fix", "resolve", "issue", "not showing", "error", "broken", "troubleshoot",
                    "discrepancy", "incorrect", "wrong", "duplicate", "stale"],
    "Update/Modify": ["update", "change", "modify", "adjust", "edit", "revise", "correct",
                      "rename", "amend"],
    "Review/Audit": ["review", "audit", "reconcile", "verify", "check", "monitor", "validate",
                     "confirm", "compare"],
    "Record/Track": ["record", "enter", "track", "log", "add entry", "capture", "note",
                     "distribution notice", "capital call"],
}


def compute_cluster_analytics(texts: list, platforms: list) -> dict:
    """Compute analytics for a cluster."""
    total = len(texts)

    # Action type breakdown
    action_counts = {}
    for action, keywords in ACTION_KEYWORDS.items():
        count = sum(1 for t in texts if any(k in t.lower() for k in keywords))
        action_counts[action] = count

    action_pcts = {k: (v / total * 100) for k, v in action_counts.items()}

    # Platform distribution
    platform_counts = pd.Series(platforms).value_counts()
    platform_pcts = {str(k): float(v / total * 100) for k, v in platform_counts.items()}

    # Complexity (by text length)
    lengths = [len(t) for t in texts]
    simple = sum(1 for l in lengths if l < 80)
    medium = sum(1 for l in lengths if 80 <= l < 300)
    complex_t = sum(1 for l in lengths if l >= 300)
    complexity = {
        "Simple (<80 chars)": simple / total * 100,
        "Medium (80-300 chars)": medium / total * 100,
        "Complex (300+ chars)": complex_t / total * 100,
    }

    # Key metrics
    word_counts = [len(t.split()) for t in texts]
    metrics = {
        "avg_words": sum(word_counts) / len(word_counts),
        "median_words": sorted(word_counts)[len(word_counts) // 2],
        "max_chars": max(lengths),
        "min_chars": min(lengths),
        "avg_chars": sum(lengths) / len(lengths),
    }

    return {
        "action_types": action_pcts,
        "platforms": platform_pcts,
        "complexity": complexity,
        "metrics": metrics,
    }


class ClusterPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.add_font("DejaVu", "", "C:/Windows/Fonts/arial.ttf", uni=True)
        self.add_font("DejaVu", "B", "C:/Windows/Fonts/arialbd.ttf", uni=True)
        self.add_font("DejaVu", "I", "C:/Windows/Fonts/ariali.ttf", uni=True)

    def header(self):
        if self.page_no() > 1:
            self.set_font("DejaVu", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 5, "Cluster Analysis Report - TOS/Elevate Advisory", align="R")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def title_page(self, total_tasks, total_clusters):
        self.add_page()
        self.ln(60)
        self.set_font("DejaVu", "B", 28)
        self.set_text_color(30, 60, 114)
        self.cell(0, 15, "Cluster Analysis Report", align="C")
        self.ln(20)
        self.set_font("DejaVu", "", 16)
        self.set_text_color(60, 60, 60)
        self.cell(0, 10, "Asana Task Classification System", align="C")
        self.ln(8)
        self.cell(0, 10, "TOS / Elevate Advisory", align="C")
        self.ln(20)
        self.set_font("DejaVu", "", 12)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, f"{total_tasks:,} Tasks | {total_clusters} Sub-type Clusters | 8 Issue Types", align="C")
        self.ln(8)
        self.cell(0, 8, "v2 Taxonomy (19 Sub-types)", align="C")
        self.ln(30)
        self.set_font("DejaVu", "I", 10)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 6, (
            "This report provides a detailed analysis of each task cluster in the "
            "classification system. For each of the 19 sub-type clusters, an AI-generated "
            "summary explains the nature and themes of the tasks, followed by analytics "
            "showing task type breakdown, platform distribution, and complexity metrics, "
            "then a complete listing of all task descriptions with Asana GIDs."
        ), align="C")

    def table_of_contents(self, clusters_by_pillar):
        self.add_page()
        self.set_font("DejaVu", "B", 18)
        self.set_text_color(30, 60, 114)
        self.cell(0, 12, "Table of Contents", align="L")
        self.ln(15)

        for pillar in PILLAR_ORDER:
            if pillar not in clusters_by_pillar:
                continue
            self.set_font("DejaVu", "B", 11)
            self.set_text_color(30, 60, 114)
            self.cell(0, 8, pillar, align="L")
            self.ln(7)

            for label, count in clusters_by_pillar[pillar]:
                self.set_font("DejaVu", "", 10)
                self.set_text_color(60, 60, 60)
                self.cell(10, 6, "")
                self.cell(0, 6, f"{label} ({count} tasks)")
                self.ln(5)
            self.ln(3)

    def analytics_section(self, analytics, total_dataset_tasks):
        """Render the analytics box between summary and task listing."""
        self.set_font("DejaVu", "B", 11)
        self.set_text_color(40, 40, 40)
        self.cell(0, 7, "Cluster Analytics")
        self.ln(7)

        # Draw a light gray background box
        box_y = self.get_y()
        self.set_fill_color(245, 247, 250)
        self.rect(10, box_y, 190, 58, "F")

        # Left column: Action Types
        self.set_xy(14, box_y + 3)
        self.set_font("DejaVu", "B", 8)
        self.set_text_color(30, 60, 114)
        self.cell(85, 4, "Action Type Breakdown")
        self.ln(5)

        self.set_font("DejaVu", "", 8)
        self.set_text_color(40, 40, 40)
        for action, pct in sorted(analytics["action_types"].items(), key=lambda x: -x[1]):
            self.set_x(14)
            bar_width = pct * 0.4  # scale to fit
            self.cell(85, 4, f"  {action}: {pct:.1f}%")
            self.ln(4)

        # Left column: Complexity
        self.ln(2)
        self.set_x(14)
        self.set_font("DejaVu", "B", 8)
        self.set_text_color(30, 60, 114)
        self.cell(85, 4, "Complexity Distribution")
        self.ln(5)

        self.set_font("DejaVu", "", 8)
        self.set_text_color(40, 40, 40)
        for level, pct in analytics["complexity"].items():
            self.set_x(14)
            self.cell(85, 4, f"  {level}: {pct:.1f}%")
            self.ln(4)

        # Right column: Platform Mix
        right_x = 110
        self.set_xy(right_x, box_y + 3)
        self.set_font("DejaVu", "B", 8)
        self.set_text_color(30, 60, 114)
        self.cell(85, 4, "Platform Distribution")
        self.ln(5)

        self.set_font("DejaVu", "", 8)
        self.set_text_color(40, 40, 40)
        for platform, pct in sorted(analytics["platforms"].items(), key=lambda x: -x[1]):
            self.set_x(right_x)
            self.cell(85, 4, f"  {platform}: {pct:.1f}%")
            self.ln(4)

        # Right column: Key Metrics
        metrics = analytics["metrics"]
        self.ln(2)
        self.set_x(right_x)
        self.set_font("DejaVu", "B", 8)
        self.set_text_color(30, 60, 114)
        self.cell(85, 4, "Key Metrics")
        self.ln(5)

        self.set_font("DejaVu", "", 8)
        self.set_text_color(40, 40, 40)
        metric_lines = [
            f"  Avg words/task: {metrics['avg_words']:.0f}",
            f"  Median words/task: {metrics['median_words']}",
            f"  Avg chars/task: {metrics['avg_chars']:.0f}",
            f"  Longest task: {metrics['max_chars']} chars",
        ]
        for line in metric_lines:
            self.set_x(right_x)
            self.cell(85, 4, line)
            self.ln(4)

        # Move below the box
        self.set_y(box_y + 62)

    def cluster_section(self, label, pillar, task_count, summary, task_ids, tasks,
                        analytics, total_dataset_tasks):
        self.add_page()

        # Cluster header
        self.set_font("DejaVu", "B", 16)
        self.set_text_color(30, 60, 114)
        self.multi_cell(0, 8, label)
        self.ln(3)

        # Pillar, count, and share
        share_pct = task_count / total_dataset_tasks * 100
        self.set_font("DejaVu", "", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, f"Issue Type: {pillar}  |  Tasks: {task_count}  |  Share of Total: {share_pct:.1f}%")
        self.ln(10)

        # Summary section
        self.set_font("DejaVu", "B", 11)
        self.set_text_color(40, 40, 40)
        self.cell(0, 7, "Cluster Summary")
        self.ln(6)

        self.set_font("DejaVu", "I", 9)
        self.set_text_color(50, 50, 50)
        clean_summary = summary.replace("\n", " ").strip()
        self.multi_cell(0, 5, clean_summary)
        self.ln(6)

        # Analytics section
        self.analytics_section(analytics, total_dataset_tasks)

        # Divider
        self.set_draw_color(180, 180, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

        # Task listing header
        self.set_font("DejaVu", "B", 11)
        self.set_text_color(40, 40, 40)
        self.cell(0, 7, f"All Tasks ({task_count})")
        self.ln(8)

        # List all tasks with GID headers
        for i, (gid, task) in enumerate(zip(task_ids, tasks), 1):
            # Truncate very long tasks to 500 chars
            display_text = task.strip()
            if len(display_text) > 500:
                display_text = display_text[:497] + "..."

            # Calculate space needed
            text = f"{i}. {display_text}"
            lines_needed = len(text) // 90 + 2  # +1 for GID line
            space_needed = lines_needed * 4 + 8
            if self.get_y() + space_needed > 275:
                self.add_page()

            # GID header (small, gray)
            self.set_font("DejaVu", "", 6.5)
            self.set_text_color(140, 140, 140)
            self.cell(0, 3, f"[GID: {gid}]")
            self.ln(3)

            # Task text
            self.set_font("DejaVu", "", 8)
            self.set_text_color(30, 30, 30)
            self.multi_cell(0, 4, text)
            self.ln(2)


def main():
    print("=" * 60)
    print("GENERATING CLUSTER ANALYSIS PDF (Enhanced)")
    print("=" * 60)

    # Load data
    df = pd.read_csv(TRAIN_PATH, dtype=str, keep_default_na=False)
    with open(SUMMARIES_PATH, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    total_tasks = len(df)
    print(f"Loaded {total_tasks} tasks, {len(summaries)} summaries")

    # Organize clusters by pillar
    clusters_by_pillar = {}
    for label, info in summaries.items():
        pillar = info["pillar"]
        if pillar not in clusters_by_pillar:
            clusters_by_pillar[pillar] = []
        clusters_by_pillar[pillar].append((label, info["task_count"]))

    # Sort within each pillar
    for pillar in clusters_by_pillar:
        clusters_by_pillar[pillar].sort(key=lambda x: x[0])

    # Create PDF
    pdf = ClusterPDF()

    # Title page
    pdf.title_page(total_tasks=total_tasks, total_clusters=len(summaries))

    # Table of contents
    pdf.table_of_contents(clusters_by_pillar)

    # Generate each cluster section (ordered by pillar)
    cluster_num = 0
    for pillar in PILLAR_ORDER:
        if pillar not in clusters_by_pillar:
            continue
        for label, count in clusters_by_pillar[pillar]:
            cluster_num += 1
            print(f"  [{cluster_num}/19] {label} ({count} tasks)")

            info = summaries[label]
            subset = df[df["label"] == label]
            task_ids = subset["task_id"].tolist()
            tasks = subset["cleaned_text"].tolist()
            platforms = subset["primary_platform"].tolist()

            # Compute analytics
            analytics = compute_cluster_analytics(tasks, platforms)

            pdf.cluster_section(
                label=label,
                pillar=pillar,
                task_count=count,
                summary=info["summary"],
                task_ids=task_ids,
                tasks=tasks,
                analytics=analytics,
                total_dataset_tasks=total_tasks,
            )

    # Save
    pdf.output(str(OUTPUT_PATH))
    file_size = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"PDF saved: {OUTPUT_PATH} ({file_size:.1f} MB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
