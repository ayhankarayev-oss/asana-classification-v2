"""
Main Preprocessing Script
=========================
Reads : Client_support.csv
Writes: outputs/corpus_clean.csv       (cleaned text for clustering + metadata)
        outputs/anonymization_log.csv  (audit trail of every entity removal)

Traceability guarantee:
  Every row has a task_id (Asana GID) that traces back to Client_support.csv.
  Run: merged = df_clusters.merge(df_raw, on='task_id')
"""
import html
import os
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of working directory
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

import pandas as pd

from pipeline.preprocessor import TaskPreprocessor
from pipeline.presidio_scanner import PresidioScanner

# ===========================================================================
# Config
# ===========================================================================
INPUT_FILE = Path("Client_support.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_CORPUS = OUTPUT_DIR / "corpus_clean.csv"
OUTPUT_LOG = OUTPUT_DIR / "anonymization_log.csv"
OUTPUT_PRESIDIO_LOG = OUTPUT_DIR / "presidio_detections.csv"
MIN_TEXT_LEN = 15

# ===========================================================================
# PII Masking patterns (applied BEFORE entity extraction)
# ===========================================================================
PII_PATTERNS = [
    (re.compile(r'[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}'), '[EMAIL]'),
    (re.compile(r'https?://\S+|www\.\S+'), '[URL]'),
    (re.compile(r'app\.asana\.com/\S*'), '[URL]'),
    (re.compile(r'(?:entity|account|owner|task|direct|id|ticket|portfolio|view)'
                r'\s*(?:id|#|:)?\s*[:=]?\s*\d{4,}', re.IGNORECASE), '[ID]'),
    (re.compile(r'\b\d{8,}\b'), '[ID]'),
    (re.compile(r'\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}'), '[PHONE]'),
]

# ===========================================================================
# Notes cleaning
# ===========================================================================
_BOILERPLATE = [
    re.compile(r'[-\u2014\u2015\u2500]{3,}\s*\n?This task was submitted.*', re.DOTALL | re.IGNORECASE),
    # Catch the boilerplate even without leading dashes
    re.compile(r'This task was submitted through[^\n]*', re.IGNORECASE),
    re.compile(r'https?://\S+'),
    re.compile(r'app\.asana\.com/\S*'),
    re.compile(r'(?:best regards|kind regards|regards,|thank you in advance)[,.\s\n].*', re.DOTALL | re.IGNORECASE),
    re.compile(r'\[insert [^\]]+\]', re.IGNORECASE),
    re.compile(r'\*{2,}'),
    re.compile(r'^\s*-{4,}\s*$', re.MULTILINE),
    # Form field example text (contains "Connery Collections" placeholder)
    re.compile(r'e\.g\.,?\s*[^.:\n]*?Connery Collections[^.:\n]*', re.IGNORECASE),
]

# Form field labels that are template boilerplate (zero operational signal).
# These get stripped from notes BEFORE building cleaned_text.
_FORM_LABELS = re.compile(
    r'^(?:'
    r'Name:|Email address:|How may we assist you\??:|Please specify\.?:|'
    r'Any additional information you would like to include\??:|'
    r'Please select the priority level (?:for|of) this request\.?:|'
    r'Please select the issue\.?:|Please select the type of account\.?:|'
    r'Please select the Sub Asset class:|Please confirm the asset class\.?:|'
    r'Commitment amount\.?:|Commitment date\.?:|Name of the investment\.?:|'
    r'Please specify\.?\s*:?|'
    r'Name of the direct owner\.?:|What is the name of the financial institution\??:|'
    r'Please provide more details on the data classification issue \(optional\)\.?:'
    r')\s*$',
    re.MULTILINE | re.IGNORECASE
)

# Form field ANSWERS that are pure metadata noise (priority levels, etc.)
_FORM_ANSWERS_NOISE = re.compile(
    r'^(?:Medium|High|Low|Normal|Urgent)\s*$',
    re.MULTILINE | re.IGNORECASE
)

# Unicode dashes used as separators in Asana form submissions
_UNICODE_SEPARATORS = re.compile(r'[\u2014\u2015\u2500\u2501\u2550]{3,}')


def clean_notes(text: str) -> str:
    """Remove HTML, boilerplate, form labels, and noise from notes."""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    for pat in _BOILERPLATE:
        text = pat.sub(' ', text)
    # Strip form template labels (they carry zero operational signal)
    text = _FORM_LABELS.sub('', text)
    text = _FORM_ANSWERS_NOISE.sub('', text)
    text = _UNICODE_SEPARATORS.sub('', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ===========================================================================
# Form field extraction
# ===========================================================================
def extract_field(notes_text: str, label: str) -> str:
    if not notes_text:
        return ""
    pat = re.compile(re.escape(label) + r'\s*\n(.*?)(?=\n[A-Z][^\n:]{2,80}[?]?:|\Z)', re.DOTALL | re.IGNORECASE)
    m = pat.search(notes_text)
    return m.group(1).strip() if m else ""


# ===========================================================================
# PII name masking (from Household/Assignee columns)
# ===========================================================================
_BUSINESS_SUFFIX = re.compile(
    r',?\s+(?:LLC|LLP|LP|Inc\.?|Ltd\.?|Corp\.?|Family\s+Office|Capital(?:\s+Partners)?|'
    r'Holdings?|Partners?|Advisors?|Ventures?|Investments?|Management|Group|Trust|Wealth)\b\.?',
    re.IGNORECASE)
_SKIP = {'the', 'and', 'for', 'llc', 'lp', 'inc', 'ltd', 'n/a', 'na', 'firm', 'group', 'trust'}


def _generate_name_variants(name: str) -> list[str]:
    """
    Generate fuzzy variants for a client/household name.
    e.g., 'PJS Family Office' -> ['PJS Family Office', 'PJS Family', 'PJS']
    e.g., 'Dorsar Investment Partners' -> ['Dorsar Investment Partners', 'Dorsar Investment', 'Dorsar']
    """
    name = name.strip()
    if not name:
        return []

    variants = set()
    variants.add(name)

    # Strip business suffix -> core
    core = _BUSINESS_SUFFIX.sub('', name).strip().strip(',').strip()
    if core and core != name and len(core) >= 3:
        variants.add(core)

    # First word alone (if it's substantial -- 4+ chars, not generic)
    words = name.split()
    if len(words) >= 2 and len(words[0]) >= 4 and words[0].lower() not in _SKIP:
        variants.add(words[0])

    # Last word alone (catches surnames like 'Schmulen' from 'Carl Schmulen')
    if len(words) >= 2 and len(words[-1]) >= 4 and words[-1].lower() not in _SKIP:
        variants.add(words[-1])

    # First two words (for names like 'Carl Schmulen')
    if len(words) >= 2:
        two = ' '.join(words[:2])
        if len(two) >= 5 and two.lower() not in _SKIP:
            variants.add(two)

    # Initials/abbreviation for 2+ word names (e.g., 'AWB' from 'AWB Capital')
    if len(words) >= 2:
        initials = ''.join(w[0].upper() for w in words if len(w) > 1 and w.lower() not in _SKIP)
        if len(initials) >= 3:
            variants.add(initials)

    return [v for v in variants if v and len(v) >= 3 and v.lower() not in _SKIP]


def build_pii_rules(df: pd.DataFrame) -> list[tuple[re.Pattern, str]]:
    """
    Build client/employee masking rules from DataFrame columns.
    Uses fuzzy variant generation to catch 'PJS', 'PJS Family', 'PJS Family Office' etc.
    """
    rules = []
    skip_hh = {'n/a', 'firm-wide', '', 'firm wide', 'epicc'}  # Epicc is a platform, not a client

    if "Household" in df.columns:
        for cell in df["Household"].unique():
            for part in str(cell).split(','):
                part = part.strip()
                if part.lower() not in skip_hh and part and len(part) >= 3:
                    for v in _generate_name_variants(part):
                        rules.append((v, '[CLIENT]'))

    if "Assignee" in df.columns:
        for cell in df["Assignee"].unique():
            cell = str(cell).strip()
            if cell and '@' not in cell and len(cell) >= 3:
                rules.append((cell, '[EMPLOYEE]'))
                # Also add first name alone if 2+ word name
                parts = cell.split()
                if len(parts) >= 2 and len(parts[0]) >= 4:
                    rules.append((parts[0], '[EMPLOYEE]'))

    # Compile longest-first, case-insensitive, whole-word
    compiled = []
    seen = set()
    for variant, ph in sorted(rules, key=lambda x: -len(x[0])):
        if variant.lower() not in seen:
            seen.add(variant.lower())
            pat = re.compile(r'\b' + re.escape(variant) + r'\b', re.IGNORECASE)
            compiled.append((pat, ph))
    return compiled


def apply_pii(text: str, pii_rules: list) -> str:
    if not text:
        return text
    for pat, replacement in PII_PATTERNS:
        text = pat.sub(f' {replacement} ', text)
    for pat, replacement in pii_rules:
        text = pat.sub(f' {replacement} ', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: '{INPUT_FILE}' not found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    n_raw = len(df)
    print(f"  {n_raw:,} rows loaded")

    # Immutable task_id from Asana GID
    df["task_id"] = df["Task ID"]
    print(f"  task_id assigned from Asana GID column")

    # --- Filter chain ---
    print("\n[FILTER]")
    mask = df["Parent task"] != ""
    df = df[~mask].copy()
    print(f"  Sub-tasks removed: -{mask.sum():,}  (remaining: {len(df):,})")

    mask = df["Name"].str.contains("KaizenBot", case=False, regex=False)
    df = df[~mask].copy()
    print(f"  KaizenBot removed: -{mask.sum():,}  (remaining: {len(df):,})")

    mask = df["Name"].str.contains("Working Session", case=False, regex=False)
    df = df[~mask].copy()
    print(f"  Working Sessions: -{mask.sum():,}  (remaining: {len(df):,})")

    mask = (df["Name"] == "") & (df["Notes"] == "")
    df = df[~mask].copy()
    print(f"  Empty rows:       -{mask.sum():,}  (remaining: {len(df):,})")

    # --- Form fields ---
    print("\n[FORM FIELDS]")
    df["form_intent"] = df["Notes"].apply(lambda t: extract_field(t, "How may we assist you?:"))
    alt = df["Notes"].apply(lambda t: extract_field(t, "Please select the issue.:"))
    df["form_intent"] = df["form_intent"].where(df["form_intent"] != "", alt)
    df["form_detail"] = df["Notes"].apply(lambda t: extract_field(t, "Please specify.:"))
    print(f"  form_intent: {(df['form_intent'] != '').sum():,} rows")

    # --- Notes cleaning ---
    df["notes_clean"] = df["Notes"].apply(clean_notes)

    # --- Build combined text ---
    def build_text(row):
        parts = []
        if row["form_intent"]:
            parts.append(row["form_intent"])
        parts.append(row["Name"])
        if row["notes_clean"]:
            parts.append(row["notes_clean"])
        return " ".join(p for p in parts if p).strip()

    df["original_text"] = df.apply(build_text, axis=1)

    # --- PII masking ---
    print("\n[PII MASKING]")
    pii_rules = build_pii_rules(df)
    print(f"  {len(pii_rules):,} PII rules compiled")
    df["text_pii_masked"] = df["original_text"].apply(lambda t: apply_pii(t, pii_rules))

    # --- Presidio ML-based PII detection ---
    print("\n[PRESIDIO SCAN]")
    print("  Initializing Presidio (this may take a moment to load models)...")
    presidio = PresidioScanner(confidence_threshold=0.85)
    
    presidio_detections = []
    presidio_masked = []
    
    print(f"  Scanning {len(df):,} texts for person names, organizations, etc...")
    for idx, row in df.iterrows():
        masked_text, detections = presidio.scan_and_mask(row["text_pii_masked"], mask_location=False)
        presidio_masked.append(masked_text)
        
        # Add task_id to each detection for audit log
        for det in detections:
            det["task_id"] = row["task_id"]
        presidio_detections.extend(detections)
        
        # Progress indicator every 500 rows
        if (idx + 1) % 500 == 0:
            print(f"    ... {idx+1:,}/{len(df):,} processed")
    
    df["text_presidio_masked"] = presidio_masked
    print(f"  Presidio detected {len(presidio_detections):,} PII entities")

    # --- Entity extraction + cleaning ---
    print("\n[ENTITY EXTRACTION + CLEANING]")
    preprocessor = TaskPreprocessor()
    processed_df, log_df = preprocessor.process_dataframe(
        df[["task_id", "text_presidio_masked"]].rename(columns={"text_presidio_masked": "text"}),
        text_col="text", id_col="task_id")

    df = df.merge(
        processed_df[["task_id", "cleaned_text", "entities", "primary_platform", "secondary_platform"]],
        on="task_id", how="left")
    print(f"  {len(df):,} tasks processed, {len(log_df):,} log entries")

    # --- Text normalization (lowercase for embedding consistency) ---
    print("\n[TEXT NORMALIZATION]")
    df["cleaned_text"] = df["cleaned_text"].str.lower()
    print(f"  Normalized {len(df):,} texts to lowercase")

    # --- Text length filter ---
    before = len(df)
    df = df[df["cleaned_text"].str.len() >= MIN_TEXT_LEN].copy()
    if before - len(df) > 0:
        print(f"  Dropped {before - len(df):,} short texts (<{MIN_TEXT_LEN} chars)")

    # --- Deduplication ---
    print("\n[DEDUP]")
    before_dedup = len(df)
    df["dup_count"] = df.groupby("cleaned_text")["cleaned_text"].transform("count")
    df = df.drop_duplicates(subset="cleaned_text", keep="first").copy()
    print(f"  {before_dedup:,} -> {len(df):,} (-{before_dedup - len(df):,} duplicates)")

    # --- Traceability check ---
    trace = preprocessor.verify_traceability(df, df, log_df)
    print(f"\n[TRACEABILITY] Log IDs valid: {trace['all_log_ids_valid']}")

    # --- Save ---
    print("\n[SAVE]")
    out_cols = ["task_id", "Name", "original_text", "cleaned_text", "entities",
                "primary_platform", "secondary_platform", "form_intent", "form_detail",
                "Household", "Assignee", "Created At", "dup_count"]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(OUTPUT_CORPUS, index=False, encoding="utf-8-sig")
    print(f"  {OUTPUT_CORPUS}: {len(df):,} rows")

    log_df.to_csv(OUTPUT_LOG, index=False, encoding="utf-8-sig")
    print(f"  {OUTPUT_LOG}: {len(log_df):,} entries")

    # Save Presidio detections log
    if presidio_detections:
        presidio_log_df = pd.DataFrame(presidio_detections)
        presidio_log_df.to_csv(OUTPUT_PRESIDIO_LOG, index=False, encoding="utf-8-sig")
        print(f"  {OUTPUT_PRESIDIO_LOG}: {len(presidio_detections):,} detections")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"DONE | {n_raw:,} raw -> {len(df):,} final")
    print(f"{'='*60}")
    print(f"\nPlatform distribution:")
    for p, c in df["primary_platform"].value_counts().items():
        print(f"  {p:<12}: {c:>5,} ({c/len(df)*100:.1f}%)")

    print(f"\nSample (3 rows):")
    for _, row in df.sample(min(3, len(df)), random_state=42).iterrows():
        print(f"  [{row['task_id']}] {row['cleaned_text'][:80]}  | platform={row['primary_platform']}")


if __name__ == "__main__":
    main()
