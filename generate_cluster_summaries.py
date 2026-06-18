"""
generate_cluster_summaries.py
==============================
Generates 6-7 sentence LLM summaries for each of the 19 clusters
using Snowflake Cortex AI (llama3.1-70b).
Saves results to outputs/cluster_summaries.json.
"""
import json
import pandas as pd
from pathlib import Path
import snowflake.connector
import os

TRAIN_PATH = Path("outputs/training_data_v2.csv")
OUTPUT_PATH = Path("outputs/cluster_summaries.json")
LABEL_MAP_PATH = Path("models/label_map_v2.json")

MAX_SAMPLE_TASKS = 40  # sample tasks to fit context window
MAX_CHARS_PER_TASK = 300  # truncate long tasks for the prompt


def get_snowflake_connection():
    """Connect using default Snowflake connection."""
    from snowflake.connector import connect
    # Use default connection from config
    conn = connect(
        connection_name="default"
    )
    return conn


def generate_summary(conn, label: str, pillar: str, tasks: list) -> str:
    """Call Cortex AI to generate cluster summary."""
    # Build task list for prompt
    task_list = ""
    for i, task in enumerate(tasks[:MAX_SAMPLE_TASKS], 1):
        truncated = task[:MAX_CHARS_PER_TASK].strip()
        if len(task) > MAX_CHARS_PER_TASK:
            truncated += "..."
        task_list += f"{i}. {truncated}\n"

    prompt = f"""You are analyzing Asana task descriptions from a Family Office data management firm (TOS/Elevate Advisory). These tasks belong to the cluster "{label}" under the Issue Type "{pillar}".

Based on the {len(tasks)} total tasks in this cluster (sample shown below), write exactly 6-7 sentences summarizing:
- What these tasks are about
- What problems they solve or what work they represent
- What common themes and patterns appear
- What platforms or systems are mentioned

Be specific and reference actual patterns from the task descriptions. Write in professional, clear language suitable for a leadership report.

Sample tasks ({min(len(tasks), MAX_SAMPLE_TASKS)} of {len(tasks)} total):
{task_list}"""

    # Escape single quotes for SQL
    prompt_escaped = prompt.replace("'", "''")

    sql = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('llama3.1-70b', '{prompt_escaped}') AS summary"

    cursor = conn.cursor()
    cursor.execute(sql)
    result = cursor.fetchone()
    cursor.close()

    return result[0].strip() if result else "Summary generation failed."


def main():
    print("=" * 60)
    print("GENERATING CLUSTER SUMMARIES VIA CORTEX AI")
    print("=" * 60)

    # Load data
    df = pd.read_csv(TRAIN_PATH, dtype=str, keep_default_na=False)
    with open(LABEL_MAP_PATH, "r") as f:
        label_to_pillar = json.load(f)

    print(f"Loaded {len(df)} tasks across {df['label'].nunique()} clusters")

    # Connect to Snowflake
    print("\nConnecting to Snowflake...")
    conn = get_snowflake_connection()
    print("Connected.")

    # Generate summaries for each cluster
    summaries = {}
    labels = sorted(df["label"].unique())

    for i, label in enumerate(labels, 1):
        pillar = label_to_pillar.get(label, "Unknown")
        tasks = df[df["label"] == label]["cleaned_text"].tolist()

        print(f"\n[{i}/19] {label} ({len(tasks)} tasks, pillar: {pillar})")
        print(f"  Calling Cortex AI...")

        summary = generate_summary(conn, label, pillar, tasks)
        summaries[label] = {
            "pillar": pillar,
            "task_count": len(tasks),
            "summary": summary,
        }
        print(f"  Done. Summary: {summary[:80]}...")

    conn.close()

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Saved {len(summaries)} summaries to {OUTPUT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
