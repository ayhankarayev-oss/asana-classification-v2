"""
Task Preprocessor - Orchestrator for entity extraction + text cleaning.

Note: The task_id is used as the immutable index throughout the pipeline to ensure
that even after cleaning and embedding, we can always retrieve the original raw text.
Under no circumstances should the task_id be dropped, modified, or re-indexed during
the TextCleaner or EntityExtractor steps. It is treated as immutable metadata that
travels alongside the text object.
"""
import json
from typing import Optional

import pandas as pd

from pipeline.extractor import EntityExtractor
from pipeline.cleaner import TextCleaner

TRACKED_PLATFORMS = {"addepar", "arch", "egnyte", "orca", "venn"}


class TaskPreprocessor:
    """Orchestrate entity extraction, text cleaning, and platform assignment."""

    def __init__(self, config_path: Optional[str] = None):
        self.extractor = EntityExtractor(config_path)
        self.cleaner = TextCleaner()

    def process(self, text: str, task_id: Optional[str] = None) -> dict:
        """
        Process a single task text.
        Returns: {task_id, original_text, cleaned_text, entities, primary_platform, secondary_platform, anonymization_log}
        """
        entities = self.extractor.extract(text)
        cleaned_text, log_entries = self.cleaner.clean(text, entities, task_id=task_id)
        primary, secondary = self._assign_platforms(entities)

        entities_clean = [{"entity": e["entity"], "category": e["category"]} for e in entities]

        return {
            "task_id": task_id,
            "original_text": text,
            "cleaned_text": cleaned_text,
            "entities": entities_clean,
            "primary_platform": primary,
            "secondary_platform": secondary,
            "anonymization_log": log_entries,
        }

    def process_dataframe(self, df: pd.DataFrame, text_col: str, id_col: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Batch-process a DataFrame. task_id is IMMUTABLE -- never modified or re-indexed.
        Returns: (processed_df, anonymization_log_df)
        """
        results = []
        all_log_entries = []

        for idx, row in df.iterrows():
            task_id = str(row[id_col]) if (id_col and id_col in df.columns) else str(idx)
            text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            result = self.process(text, task_id=task_id)

            results.append({
                "task_id": result["task_id"],
                "original_text": result["original_text"],
                "cleaned_text": result["cleaned_text"],
                "entities": json.dumps(result["entities"]),
                "primary_platform": result["primary_platform"],
                "secondary_platform": result["secondary_platform"],
            })
            all_log_entries.extend(result["anonymization_log"])

        processed_df = pd.DataFrame(results)

        original_cols = [c for c in df.columns if c not in processed_df.columns]
        if original_cols:
            extra = df[original_cols].copy()
            extra.index = range(len(extra))
            processed_df = pd.concat([processed_df, extra], axis=1)

        log_df = pd.DataFrame(all_log_entries) if all_log_entries else pd.DataFrame(
            columns=["task_id", "original_text", "entity_removed", "category",
                     "span_start", "span_end", "context_window", "cleaned_text", "flags"])

        return processed_df, log_df

    def _assign_platforms(self, entities: list[dict]) -> tuple[str, Optional[str]]:
        """Assign primary/secondary platform from extracted entities (tracked only)."""
        # Normalize to canonical form: Addepar, Arch, Egnyte, Orca, Venn
        canonical = {"addepar": "Addepar", "arch": "Arch", "egnyte": "Egnyte", "orca": "Orca", "venn": "Venn"}
        tracked_found = []
        for e in entities:
            key = e["entity"].lower()
            if key in TRACKED_PLATFORMS:
                tracked_found.append(canonical[key])
        if not tracked_found:
            return "General", None
        elif len(tracked_found) == 1:
            return tracked_found[0], None
        else:
            return tracked_found[0], tracked_found[1]

    def verify_traceability(self, input_df: pd.DataFrame, output_df: pd.DataFrame, log_df: pd.DataFrame) -> dict:
        """Post-processing check: verify ID integrity and row count alignment."""
        output_ids = set(output_df["task_id"].tolist())
        log_ids = set(log_df["task_id"].tolist()) if not log_df.empty else set()
        orphaned = log_ids - output_ids
        return {
            "input_rows": len(input_df),
            "output_rows": len(output_df),
            "rows_match": len(input_df) == len(output_df),
            "all_ids_present_in_output": len(output_ids) == len(output_df),
            "all_log_ids_valid": len(orphaned) == 0,
            "orphaned_log_ids": list(orphaned),
        }
