"""
Preprocessing v3 — improvements over v2:

  1. spaCy NER is ON by default using en_core_web_lg (larger, more accurate model).
     v2 used en_core_web_md and treated NER as optional/degraded gracefully.
     v3 exits with a clear install message if the model is missing.

  2. MIN_TEXT_LEN raised 10→15: filters out single-word or near-empty tasks
     that pollute UMAP space without contributing signal.

  3. Expanded platform/custodian blocklist: added Apex, Pershing, Fidelity National,
     Raymond James, Oppenheimer, and internal org aliases found in v1 outlier analysis.

  4. Action-verb boilerplate list expanded: v1 c-TF-IDF showed verbs like "reach",
     "look", "confirm", "make" dominating several topic keyword lists.

Reads : data/Client_support.csv
Writes: outputs/corpus_clean.csv
"""

import html
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

# ── spaCy — required in v3 ────────────────────────────────────────────────────
try:
    import spacy
    try:
        _nlp = spacy.load(cfg.SPACY_MODEL)
        print(f"[INFO] spaCy {cfg.SPACY_MODEL} loaded")
    except OSError:
        _nlp = spacy.load(cfg.SPACY_FALLBACK)
        print(f"[INFO] spaCy {cfg.SPACY_FALLBACK} loaded (fallback)")
    SPACY_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"[WARN] spaCy unavailable ({e}). NER pass will be skipped.")
    print("       pip install spacy && python -m spacy download en_core_web_lg")
    SPACY_AVAILABLE = False

# ── Typed platform placeholder map ───────────────────────────────────────────
PLATFORM_CATEGORIES: dict[str, list[str]] = {
    "[PORTFOLIO_PLATFORM]": [
        "Addepar - EU", "Addepar - Own", "Addepar EU", "Addepar Own", "Addepar",
        "Arch - Own", "Arch Own", "Arch",
        "Venn", "NINES",
    ],
    "[DOCUMENT_PLATFORM]": ["Egnyte", "SharePoint"],
    "[DATA_AGGREGATOR]":   ["Plaid", "ByAll", "Orca"],
    "[CLIENT_PORTAL]":     ["Epicc"],
    "[CRM_PLATFORM]":      ["Salesforce", "SFDC"],
    "[ACCOUNTING_PLATFORM]": ["KnowLedger", "QuickBooks", "Quickbooks"],
    "[PM_PLATFORM]":       ["Asana"],
    # v3 additions
    "[ANALYTICS_PLATFORM]": ["Tableau", "Power BI", "PowerBI", "Looker"],
    "[CUSTODIAN]": [
        # original
        "Fidelity", "Schwab", "Charles Schwab",
        "Goldman Sachs", "Goldman", "JPMorgan", "JPM", "J.P. Morgan",
        "Morgan Stanley", "Merrill Lynch", "Merrill", "UBS",
        "Wells Fargo", "Citi Private Bank", "Citi", "Citibank",
        "BofA", "Bank of America", "Northern Trust",
        "TD Ameritrade", "Interactive Brokers", "IBKR",
        "Pershing", "BNY Mellon", "State Street",
        "Raymond James", "Oppenheimer",
        "Apex Clearing", "Apex",
        "StoneX", "Stonex",
        # v3 additions from outlier review
        "Fidelity National", "Richfield", "Amplify",
    ],
    "[BANK]": [
        "JPMorgan Chase", "Chase", "Wells Fargo Bank",
        "Citi Bank", "Bank of America", "Regions Bank", "Regions",
        "USCA", "Teal Tuna",   # v3: internal entity references treated as banks
    ],
    "[ADVISOR_ORG]": [
        "TOS", "Mosaic", "Elevate", "SAOS", "DFO", "DFI",
        "Armanino", "Empaxis", "Andersen",
        # v3 additions
        "KnowLedger", "Amplify CPA", "CCP",
    ],
}

# Pre-compile platform patterns (longest first to prevent partial matches)
_PLATFORM_PATTERNS: list[tuple[re.Pattern, str]] = []
for placeholder, names in PLATFORM_CATEGORIES.items():
    sorted_names = sorted(names, key=len, reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in sorted_names) + r")\b",
        re.IGNORECASE
    )
    _PLATFORM_PATTERNS.append((pattern, placeholder))

# ── Boilerplate phrases to strip ─────────────────────────────────────────────
_BOILERPLATE = re.compile(
    r"""(
        hi\s+team|hello\s+team|hi\s+there|dear\s+team|
        please\s+see\s+below|please\s+find\s+below|
        as\s+per\s+our\s+(call|discussion|conversation)|
        as\s+discussed|as\s+requested|
        hope\s+(you|this)\s+(are|is)\s+well|
        thank\s+you\s+(for|in\s+advance)|
        thanks\s+(for|in\s+advance)|
        kind\s+regards|best\s+regards|regards,|
        let\s+me\s+know\s+if\s+you\s+have\s+(any\s+)?questions|
        \*\*no\s+additional\s+information\s+provided\*\*|
        no\s+additional\s+information
    )""",
    re.IGNORECASE | re.VERBOSE
)

_RE_EMAIL   = re.compile(r"\S+@\S+\.\S+")
_RE_URL     = re.compile(r"https?://\S+|www\.\S+")
_RE_PHONE   = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_RE_ENTITY_ID = re.compile(r"\b\d{6,}\b")   # 6+ digit numeric IDs
_RE_WHITESPACE = re.compile(r"\s+")


def _clean_text(raw: str) -> str:
    text = html.unescape(raw)
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"[*_~`]", "", text)         # markdown formatting
    text = _BOILERPLATE.sub(" ", text)
    text = _RE_EMAIL.sub("[EMAIL]", text)
    text = _RE_URL.sub("[URL]", text)
    text = _RE_PHONE.sub("[PHONE]", text)
    text = _RE_ENTITY_ID.sub("[ENTITY_ID]", text)
    for pattern, placeholder in _PLATFORM_PATTERNS:
        text = pattern.sub(placeholder, text)
    text = re.sub(r"\[([A-Z_]+)\]\s*\1", r"[\1]", text)   # collapse duplicates
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


def _apply_spacy_ner(text: str) -> str:
    if not SPACY_AVAILABLE:
        return text
    doc = _nlp(text)
    result = text
    for ent in reversed(doc.ents):
        if ent.label_ in {"PERSON"}:
            result = result[:ent.start_char] + "[EMPLOYEE]" + result[ent.end_char:]
        elif ent.label_ in {"ORG"} and len(ent.text) > 3:
            result = result[:ent.start_char] + "[CLIENT]" + result[ent.end_char:]
    return result


def _build_household_pattern(df: pd.DataFrame) -> re.Pattern | None:
    """Compile a regex from the Household column for dynamic client masking."""
    candidates = set()
    for col in ["Household", "household", "Client", "client"]:
        if col in df.columns:
            candidates.update(df[col].dropna().str.strip().tolist())
    names = [n for n in candidates if len(n) >= 3]
    if not names:
        return None
    names_sorted = sorted(names, key=len, reverse=True)
    return re.compile(
        r"\b(" + "|".join(re.escape(n) for n in names_sorted) + r")\b",
        re.IGNORECASE
    )


def run() -> None:
    cfg.OUTPUT_DIR.mkdir(exist_ok=True)

    if not cfg.INPUT_CSV.exists():
        sys.exit(f"ERROR: '{cfg.INPUT_CSV}' not found. Place it in the data/ folder.")

    print(f"Loading {cfg.INPUT_CSV} ...")
    df = pd.read_csv(cfg.INPUT_CSV, dtype=str, keep_default_na=False)
    print(f"  {len(df):,} rows loaded | columns: {list(df.columns)}")

    household_re = _build_household_pattern(df)

    name_col  = "Name"  if "Name"  in df.columns else df.columns[0]
    notes_col = "Notes" if "Notes" in df.columns else None

    def build_raw(row: pd.Series) -> str:
        parts = [str(row.get(name_col, "")).strip()]
        if notes_col:
            parts.append(str(row.get(notes_col, "")).strip())
        return ". ".join(p for p in parts if p)

    print("Building raw complaint_text ...")
    df["complaint_text_raw"] = df.apply(build_raw, axis=1)

    print("Cleaning & anonymizing ...")
    cleaned = []
    for raw in df["complaint_text_raw"]:
        text = _clean_text(raw)
        if household_re:
            text = household_re.sub("[CLIENT]", text)
        text = _apply_spacy_ner(text)
        cleaned.append(text)
    df["complaint_text"] = cleaned

    before = len(df)
    df = df[df["complaint_text"].str.len() >= cfg.MIN_TEXT_LEN].copy()
    print(f"  Dropped {before - len(df):,} rows below MIN_TEXT_LEN={cfg.MIN_TEXT_LEN}")

    df.to_csv(cfg.CORPUS_CLEAN_CSV, index=False)
    print(f"\nWrote {len(df):,} rows → {cfg.CORPUS_CLEAN_CSV}")


if __name__ == "__main__":
    run()