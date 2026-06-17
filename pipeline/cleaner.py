"""
Text Cleaner - Remove detected entities from text for clustering.

Strategy:
  - Remove entity text by character span, collapse whitespace.
  - Clean orphaned prepositions/conjunctions at sentence boundaries.
  - No placeholder tokens -- just clean removal.
  - Generate anonymization log entries with quality flags for auditability.

Flags:
  - trailing_preposition: removal left orphaned "in", "to", "from", "with" at end
  - short_result: cleaned text is <10 chars (may have lost too much context)
  - adjacent_removal: two entities removed within 3 words of each other
  - numeric_adjacent: entity was adjacent to a number (could be error code/version)
"""
import re
from typing import Optional

_TRAILING_PREPS = re.compile(
    r"\s+(?:in|to|from|with|at|on|for|by|of|via|into|onto|through)\s*$", re.IGNORECASE)
_LEADING_PREPS = re.compile(
    r"^\s*(?:in|to|from|with|at|on|for|by|of|via|into|onto|through)\s+", re.IGNORECASE)
_ORPHANED_CONJUNCTIONS = re.compile(r"\s+(?:and|or|&)\s*$", re.IGNORECASE)
_LEADING_CONJUNCTIONS = re.compile(r"^\s*(?:and|or|&)\s+", re.IGNORECASE)
_MULTI_SPACE = re.compile(r"\s{2,}")
_NUMERIC_ADJACENT = re.compile(r"\d")


class TextCleaner:
    """Remove entities from text and produce an audit trail."""

    def clean(self, text: str, entities: list[dict], task_id: Optional[str] = None) -> tuple[str, list[dict]]:
        """
        Remove detected entities from text.
        Returns (cleaned_text, log_entries).
        """
        if not entities:
            return text.strip(), []

        log_entries = []
        sorted_entities = sorted(entities, key=lambda e: e["span"][0], reverse=True)

        result = text
        for entity_info in sorted_entities:
            start, end = entity_info["span"]
            ctx_start = max(0, start - 10)
            ctx_end = min(len(result), end + 10)
            context_before = result[ctx_start:ctx_end]

            result = result[:start] + result[end:]

            flags = self._detect_flags(text, (start, end), entity_info["entity"], entities)
            log_entries.append({
                "task_id": task_id,
                "original_text": text,
                "entity_removed": entity_info["entity"],
                "category": entity_info["category"],
                "span_start": start,
                "span_end": end,
                "context_window": context_before,
                "flags": "|".join(flags) if flags else "",
            })

        result = self._cleanup(result)

        for entry in log_entries:
            entry["cleaned_text"] = result

        if len(result.strip()) < 10:
            for entry in log_entries:
                if "short_result" not in entry["flags"]:
                    entry["flags"] = (entry["flags"] + "|short_result" if entry["flags"] else "short_result")

        log_entries.reverse()
        return result, log_entries

    def _cleanup(self, text: str) -> str:
        """Normalize whitespace, remove orphaned prepositions/conjunctions."""
        text = _MULTI_SPACE.sub(" ", text)
        text = _TRAILING_PREPS.sub("", text)
        text = _LEADING_PREPS.sub("", text)
        text = _ORPHANED_CONJUNCTIONS.sub("", text)
        text = _LEADING_CONJUNCTIONS.sub("", text)
        text = _MULTI_SPACE.sub(" ", text)
        text = re.sub(r"\s*-\s*-\s*", " - ", text)
        text = re.sub(r",\s*,", ",", text)
        return text.strip()

    def _detect_flags(self, original_text: str, entity_span: tuple, entity_text: str, all_entities: list) -> list[str]:
        """Detect quality flags for a given entity removal."""
        flags = []
        start, end = entity_span

        pre_text = original_text[:start].rstrip()
        if re.search(r"\b(?:in|to|from|with|at|on|for|by|of|via)\s*$", pre_text, re.IGNORECASE):
            flags.append("trailing_preposition")

        pre_chars = original_text[max(0, start - 5):start]
        post_chars = original_text[end:min(len(original_text), end + 5)]
        if _NUMERIC_ADJACENT.search(pre_chars) or _NUMERIC_ADJACENT.search(post_chars):
            flags.append("numeric_adjacent")

        for other in all_entities:
            if other["span"] == entity_span:
                continue
            other_start, other_end = other["span"]
            if other_end <= start:
                gap = original_text[other_end:start]
            elif other_start >= end:
                gap = original_text[end:other_start]
            else:
                continue
            if len(gap.split()) <= 3:
                flags.append("adjacent_removal")
                break

        return flags
