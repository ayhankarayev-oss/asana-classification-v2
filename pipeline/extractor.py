"""
Entity Extractor - Deterministic regex-based entity extraction from raw text.

Design:
  - Patterns compiled longest-first to prevent partial matches.
  - Case-insensitive with word boundaries.
  - "Arch" has negative lookahead to avoid "architecture", "archive", etc.
  - Overlapping spans deduplicated (longest match wins).
"""
import re
from pathlib import Path
from typing import Optional

from pipeline.config import load_entities


class EntityExtractor:
    """Extract monitored entities from text using compiled regex patterns."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = str(Path(__file__).parent / "entities.yaml")
        self._config_path = config_path
        self._entities = self._load_config()
        self._pattern = self._compile_pattern()

    def _load_config(self) -> list[dict]:
        """Load config and flatten into [{name, category}, ...]."""
        raw = load_entities(self._config_path)
        entities = []
        for subcategory, names in raw.get("platforms", {}).items():
            for name in names:
                entities.append({"name": name, "category": f"platforms.{subcategory}"})
        for category in ("custodians", "banks", "advisor_orgs", "clients",
                         "systems", "client_codes", "banks_additional"):
            for name in raw.get(category, []):
                entities.append({"name": name, "category": category})
        return entities

    def _compile_pattern(self) -> re.Pattern:
        """Build single regex alternation, longest-first, with Arch protection."""
        sorted_entities = sorted(self._entities, key=lambda e: len(e["name"]), reverse=True)
        alt_parts = []
        for entity in sorted_entities:
            name = entity["name"]
            escaped = re.escape(name)
            if name.lower() == "arch":
                alt_parts.append(r"\b" + escaped + r"(?![a-zA-Z])")
            else:
                alt_parts.append(r"\b" + escaped + r"\b")
        pattern_str = "(?:" + "|".join(alt_parts) + ")"
        return re.compile(pattern_str, re.IGNORECASE)

    def _build_name_to_category(self) -> dict[str, str]:
        """Lowercase name -> category lookup."""
        return {e["name"].lower(): e["category"] for e in self._entities}

    def extract(self, text: str) -> list[dict]:
        """
        Extract all monitored entities from text.
        Returns list of dicts sorted by position:
        [{"entity": "Addepar", "category": "platforms.portfolio", "span": (0, 7)}, ...]
        """
        if not text or not text.strip():
            return []

        name_to_category = self._build_name_to_category()
        raw_matches = []
        for match in self._pattern.finditer(text):
            matched_text = match.group()
            category = name_to_category.get(matched_text.lower(), "unknown")
            raw_matches.append({
                "entity": matched_text,
                "category": category,
                "span": (match.start(), match.end()),
            })

        if not raw_matches:
            return []

        # Deduplicate overlapping spans (keep longest)
        raw_matches.sort(key=lambda m: (m["span"][0], -(m["span"][1] - m["span"][0])))
        deduped = []
        last_end = -1
        for m in raw_matches:
            start, end = m["span"]
            if start >= last_end:
                deduped.append(m)
                last_end = end
        return deduped

    @property
    def entity_names(self) -> list[str]:
        return [e["name"] for e in self._entities]

    @property
    def tracked_platforms(self) -> list[str]:
        """The 5 tracked platforms for the platform dimension."""
        return ["Addepar", "Arch", "Egnyte", "Orca", "Venn"]
