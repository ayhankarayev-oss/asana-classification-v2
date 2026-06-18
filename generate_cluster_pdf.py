"""
generate_cluster_pdf.py
========================
Generates a PDF report analyzing all 19 clusters with:
- LLM-generated 6-7 sentence summary per cluster
- Full listing of all tasks within each cluster
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


class ClusterPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        # Register DejaVu for Unicode support
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
            "summary explains the nature and themes of the tasks, followed by a complete "
            "listing of all task descriptions belonging to that cluster."
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

    def cluster_section(self, label, pillar, task_count, summary, tasks):
        self.add_page()

        # Cluster header
        self.set_font("DejaVu", "B", 16)
        self.set_text_color(30, 60, 114)
        self.multi_cell(0, 8, label)
        self.ln(3)

        # Pillar and count
        self.set_font("DejaVu", "", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, f"Issue Type: {pillar}  |  Tasks: {task_count}")
        self.ln(10)

        # Summary section
        self.set_font("DejaVu", "B", 11)
        self.set_text_color(40, 40, 40)
        self.cell(0, 7, "Cluster Summary")
        self.ln(6)

        self.set_font("DejaVu", "I", 9)
        self.set_text_color(50, 50, 50)
        # Clean summary text
        clean_summary = summary.replace("\n", " ").strip()
        self.multi_cell(0, 5, clean_summary)
        self.ln(8)

        # Divider
        self.set_draw_color(180, 180, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

        # Task listing header
        self.set_font("DejaVu", "B", 11)
        self.set_text_color(40, 40, 40)
        self.cell(0, 7, f"All Tasks ({task_count})")
        self.ln(8)

        # List all tasks
        self.set_font("DejaVu", "", 8)
        self.set_text_color(30, 30, 30)

        for i, task in enumerate(tasks, 1):
            # Truncate very long tasks to 500 chars
            display_text = task.strip()
            if len(display_text) > 500:
                display_text = display_text[:497] + "..."

            # Number prefix
            prefix = f"{i}. "
            text = prefix + display_text

            # Check if we need a new page
            lines_needed = len(text) // 90 + 1
            space_needed = lines_needed * 4 + 3
            if self.get_y() + space_needed > 275:
                self.add_page()
                self.set_font("DejaVu", "", 8)
                self.set_text_color(30, 30, 30)

            self.multi_cell(0, 4, text)
            self.ln(2)


def main():
    print("=" * 60)
    print("GENERATING CLUSTER ANALYSIS PDF")
    print("=" * 60)

    # Load data
    df = pd.read_csv(TRAIN_PATH, dtype=str, keep_default_na=False)
    with open(SUMMARIES_PATH, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    print(f"Loaded {len(df)} tasks, {len(summaries)} summaries")

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
    pdf.title_page(total_tasks=len(df), total_clusters=len(summaries))

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
            tasks = df[df["label"] == label]["cleaned_text"].tolist()
            pdf.cluster_section(
                label=label,
                pillar=pillar,
                task_count=count,
                summary=info["summary"],
                tasks=tasks,
            )

    # Save
    pdf.output(str(OUTPUT_PATH))
    file_size = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"PDF saved: {OUTPUT_PATH} ({file_size:.1f} MB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
