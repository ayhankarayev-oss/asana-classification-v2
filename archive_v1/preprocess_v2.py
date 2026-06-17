"""
Phase 2v2 - Text Preprocessing with Entity Anonymization
=========================================================
Reads : Client_support.csv
Writes: corpus_clean.csv   (drop-in replacement for preprocess.py output)

Why v2?
-------
The original pipeline embeds client names, platform names, and employee
names into complaint_text, causing BERTopic to cluster around WHO/WHAT
rather than around the operational issue being described.

Root causes confirmed in corpus analysis:
  - "addepar" in 1,766 texts (41%)  -->  platform name drowning issue signal
  - "arch"    in 1,576 texts (37%)  -->  same
  - "dorsar"  in 559   texts (13%)  -->  single household creating pseudo-cluster
  - 99.8% of rows contain person-name patterns (contacts embedded in task names)

Masking strategy (layered, deterministic):
  Layer 1 - Regex:    emails, URLs, numeric IDs, phone numbers
  Layer 2 - Blocklist: dynamic from Household + Assignee columns
                        typed platform categories (preserves workflow context)
  Layer 3 - spaCy NER (optional): residual PERSON / ORG entities missed
                                   by layers 1-2

Typed platform placeholders (preserve system-class context for clustering):
  [PORTFOLIO_PLATFORM]  - Addepar, Arch, Venn, NINES
  [DOCUMENT_PLATFORM]   - Egnyte
  [DATA_AGGREGATOR]     - Plaid, Orca, ByAll
  [CLIENT_PORTAL]       - Epicc
  [CRM_PLATFORM]        - Salesforce, SFDC
  [ACCOUNTING_PLATFORM] - KnowLedger, QuickBooks
  [PM_PLATFORM]         - Asana
  [CUSTODIAN]           - Fidelity, Schwab, Goldman, JPM, Merrill, UBS …
  [BANK]                - JPMorgan, Wells Fargo, Citi, BofA, Regions …
  [ADVISOR_ORG]         - TOS, Mosaic, Elevate (internal firm references)

Entity placeholders:
  [CLIENT]     - household / client / entity name (Dorsar, Goradia …)
  [EMPLOYEE]   - internal staff name              (Ricardo, Suzanne …)
  [EMAIL]      - email address
  [ENTITY_ID]  - numeric identifier (entity ID, account ID, task ID)
  [PHONE]      - phone number
  [URL]        - web link

Why typed is better than generic [PLATFORM]:
  "error in [PORTFOLIO_PLATFORM]"  clusters with valuation/data tasks
  "access to [CLIENT_PORTAL]"      clusters with portal/onboarding tasks
  "statement from [CUSTODIAN]"     clusters with account/reconciliation tasks
  Generic [PLATFORM] would merge all three into one undifferentiated group.

Output columns are identical to preprocess.py so model_pipeline.py requires
no changes.  An extra column 'complaint_text_raw' preserves the unmasked
version for audit/comparison.
"""

import re
import html
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# ===========================================================================
# 0. Optional spaCy — graceful degradation if not installed
# ===========================================================================
try:
    import spacy
    _nlp = spacy.load("en_core_web_md")
    SPACY_AVAILABLE = True
    print("[INFO] spaCy en_core_web_md loaded -- NER residual pass enabled")
except ImportError:
    SPACY_AVAILABLE = False
    print("[INFO] spaCy not installed  -- NER pass skipped")
    print("       Install: pip install spacy && python -m spacy download en_core_web_md")
except OSError:
    try:
        _nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
        print("[INFO] spaCy en_core_web_sm loaded (fallback) -- NER residual pass enabled")
    except OSError:
        SPACY_AVAILABLE = False
        print("[INFO] spaCy model not found -- NER pass skipped")
        print("       Install model: python -m spacy download en_core_web_md")

# ===========================================================================
# 1. Config
# ===========================================================================
INPUT_FILE   = Path("Client_support.csv")
OUTPUT_FILE  = Path("corpus_clean.csv")
MIN_TEXT_LEN = 10

PH = {
    # Typed platform categories — preserve system-class context
    "portfolio_platform":  "[PORTFOLIO_PLATFORM]",
    "document_platform":   "[DOCUMENT_PLATFORM]",
    "data_aggregator":     "[DATA_AGGREGATOR]",
    "client_portal":       "[CLIENT_PORTAL]",
    "crm_platform":        "[CRM_PLATFORM]",
    "accounting_platform": "[ACCOUNTING_PLATFORM]",
    "pm_platform":         "[PM_PLATFORM]",
    "custodian":           "[CUSTODIAN]",
    "bank":                "[BANK]",
    "advisor_org":         "[ADVISOR_ORG]",
    # Entity categories
    "client":    "[CLIENT]",
    "employee":  "[EMPLOYEE]",
    "email":     "[EMAIL]",
    "entity_id": "[ENTITY_ID]",
    "phone":     "[PHONE]",
    "url":       "[URL]",
}

# ===========================================================================
# 2. Typed platform categories
# ===========================================================================
# Each key is the placeholder written into complaint_text.
# Sorted longest-first within each list to prevent partial shadowing
# ("Addepar - EU" must be tried before "Addepar").
#
# WHY TYPED?
# "error in [PORTFOLIO_PLATFORM]"  -> clusters with data/valuation tasks
# "access to [CLIENT_PORTAL]"      -> clusters with onboarding/portal tasks
# "statement from [CUSTODIAN]"     -> clusters with account/reconciliation tasks
# Generic [PLATFORM] would merge all three into one undifferentiated group.
PLATFORM_CATEGORIES: dict[str, list[str]] = {

    # Portfolio reporting, analytics, valuation platforms
    # Typical task context: "update in Addepar", "Arch valuations", "data fix"
    "[PORTFOLIO_PLATFORM]": [
        "Addepar - EU", "Addepar - Own", "Addepar EU", "Addepar Own", "Addepar",
        "Arch - Own", "Arch Own", "Arch",
        "Venn",
        "NINES",
    ],

    # Document storage and file management
    # Typical task context: "upload to Egnyte", "share document", "find file"
    "[DOCUMENT_PLATFORM]": [
        "Egnyte",
        "SharePoint",
    ],

    # Data aggregation / bank-feed connectors
    # Typical task context: "bank feed via Plaid", "fix Orca connection"
    "[DATA_AGGREGATOR]": [
        "Plaid",
        "ByAll",
        "Orca",
    ],

    # Client-facing portal / access systems
    # Typical task context: "EPICC login", "portal access", "grant access"
    "[CLIENT_PORTAL]": [
        "Epicc",
    ],

    # CRM platforms
    # Typical task context: "update Salesforce", "create CRM record"
    "[CRM_PLATFORM]": [
        "Salesforce", "SFDC",
    ],

    # Accounting and bookkeeping
    # Typical task context: "reconcile in QuickBooks", "KnowLedger entry"
    "[ACCOUNTING_PLATFORM]": [
        "KnowLedger",
        "QuickBooks", "Quickbooks",
    ],

    # Project / task management (internal)
    # Typical task context: "Asana task", "create in Asana"
    "[PM_PLATFORM]": [
        "Asana",
    ],

    # Custodians and brokerages (hold custody of investment accounts)
    # Typical task context: "new Fidelity account", "Schwab statement",
    #                        "transfer to Goldman", "Merrill position"
    "[CUSTODIAN]": [
        "Fidelity Investments", "Fidelity",
        "Charles Schwab", "Schwab",
        "Goldman Sachs", "Goldman",
        "Morgan Stanley",
        "Northern Trust",
        "Pershing",
        "BlackRock",
        "Vanguard",
        "Stifel",
        "Interactive Brokers", "IBKR",
        "Raymond James",
        "Edward Jones",
        "Merrill Lynch", "Merrill",
        "UBS",
        "DTC", "DTCC",
        "E*Trade", "ETrade",
    ],

    # Banking institutions (deposit accounts, wires, mortgages)
    # Typical task context: "wire from JPM", "Citi bank account",
    #                        "Wells Fargo mortgage", "link bank"
    "[BANK]": [
        "JPMorgan Chase", "JPMorgan", "JP Morgan", "JPM",
        "Wells Fargo",
        "Citibank", "Citigroup", "Citi",
        "Bank of America", "BofA",
        "US Bank", "USBank",
        "TD Ameritrade", "TD Bank",
        "Regions Bank",
    ],

    # Internal advisory firm references (TOS itself and outsourced partners)
    # Masked so clustering doesn't fragment on firm name rather than task type
    "[ADVISOR_ORG]": [
        "Mosaic Advisors", "Mosaic",
        "Elevate Advisory", "Elevate",
        "TOS",
        "SAOS",
    ],
}

# Flat lowercase set — used by spaCy pass to avoid re-masking
_PLATFORM_LOWER: set[str] = {
    term.lower()
    for terms in PLATFORM_CATEGORIES.values()
    for term in terms
}

# ===========================================================================
# 3. Variant generation
# ===========================================================================
_BUSINESS_SUFFIX_RE = re.compile(
    r',?\s+(?:LLC|L\.L\.C\.?|LLP|LP|L\.P\.?|Inc\.?|Ltd\.?|Limited|Corp\.?|'
    r'Corporation|P\.C\.?|PLLC|'
    r'Family\s+Office|Capital(?:\s+Partners)?|Holdings?|Partners?|'
    r'Advisors?|Advisory|Ventures?|Investments?|Investment\s+Partners?|'
    r'Management|Entities|Group|Trust|Wealth|Exploration|Ranch|Equities|'
    r'International|Enterprises?)\b\.?',
    re.IGNORECASE,
)

# Tokens too generic to use as standalone masks
_SKIP_TOKENS = {
    'the', 'and', 'for', 'llc', 'lp', 'inc', 'ltd', 'n/a', 'na',
    'firm', 'group', 'trust', 'cap', 'mgmt', 'co', 'int', 'intl', 'corp',
    'new', 'old', 'all', 'any', 'one', 'two', 'three', 'own',
    # 2-char abbreviations that risk false positives
    'hm', 'fj', 'w3',
}

def _make_variants(name: str) -> list[str]:
    """
    Return all surface-form variants for an entity name, longest first.
    Only variants >= 3 chars and not in the skip-token list are returned.
    """
    name = name.strip()
    if not name:
        return []

    candidates = {name}

    # Strip business suffix → core identifier
    core = _BUSINESS_SUFFIX_RE.sub('', name).strip().strip(',').strip()
    if core and core != name:
        candidates.add(core)

    # Abbreviation from initials of name words
    words = [w for w in name.split() if len(w) > 1 and w.lower() not in _SKIP_TOKENS]
    if len(words) >= 2:
        abbrev = ''.join(w[0].upper() for w in words)
        if len(abbrev) >= 3:
            candidates.add(abbrev)

    # Core initials
    core_words = [w for w in core.split() if len(w) > 1 and w.lower() not in _SKIP_TOKENS]
    if len(core_words) >= 2:
        core_abbrev = ''.join(w[0].upper() for w in core_words)
        if len(core_abbrev) >= 3:
            candidates.add(core_abbrev)

    # Punctuation-free variant (handles "Addepar - EU" → "Addepar  EU")
    no_punct = re.sub(r'[^\w\s]', '', name).strip()
    if no_punct:
        candidates.add(no_punct)

    # Filter: min length 3, not in skip list, not a pure number
    result = []
    for c in candidates:
        c = c.strip()
        if (c
                and len(c) >= 3
                and c.lower() not in _SKIP_TOKENS
                and not c.isdigit()):
            result.append(c)

    # Longest first to prevent partial shadowing
    return sorted(result, key=len, reverse=True)


def build_rules(entity_variant_pairs: list[tuple[str, str]]) -> list[tuple[re.Pattern, str]]:
    """
    Compile (regex_pattern, placeholder) pairs sorted longest-term first.
    All matches are case-insensitive and word-boundary anchored.
    """
    compiled = []
    for variant, placeholder in entity_variant_pairs:
        escaped = re.escape(variant)
        pat = re.compile(r'\b' + escaped + r'\b', re.IGNORECASE)
        compiled.append((pat, placeholder, len(variant)))
    # Sort by term length so 'Addepar - EU' matches before 'Addepar'
    compiled.sort(key=lambda x: -x[2])
    return [(pat, ph) for pat, ph, _ in compiled]


# ===========================================================================
# 4. Structural regex masks (emails, IDs, URLs, phones)
# ===========================================================================
STRUCTURAL_MASKS = [
    # Email before URL so 'user@domain.com' isn't partially matched by URL rule
    (re.compile(r'[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}'),                  PH["email"]),
    # Full URLs
    (re.compile(r'https?://\S+|www\.\S+'),                                 PH["url"]),
    # Residual Asana deep-links without scheme
    (re.compile(r'app\.asana\.com/\S*'),                                   PH["url"]),
    # Labelled IDs: "Entity ID: 12345678", "Direct Owner ID: 15502323"
    (re.compile(
        r'(?:entity|account|owner|task|direct|id|ticket|portfolio|view)'
        r'\s*(?:id|#|:)?\s*[:=]?\s*\d{4,}', re.IGNORECASE),              PH["entity_id"]),
    # Bare long numbers (8+ digits) — account/task IDs
    (re.compile(r'\b\d{8,}\b'),                                            PH["entity_id"]),
    # Phone: (xxx) xxx-xxxx  or  xxx-xxx-xxxx  or  xxx.xxx.xxxx
    (re.compile(r'\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}'),                  PH["phone"]),
]


# ===========================================================================
# 5. Core masking function
# ===========================================================================
_REPEATED_PH = re.compile(
    r'(\[(?:'
    r'PORTFOLIO_PLATFORM|DOCUMENT_PLATFORM|DATA_AGGREGATOR|CLIENT_PORTAL|'
    r'CRM_PLATFORM|ACCOUNTING_PLATFORM|PM_PLATFORM|CUSTODIAN|BANK|ADVISOR_ORG|'
    r'CLIENT|EMPLOYEE|EMAIL|ENTITY_ID|PHONE|URL'
    r')\])(?:\s+\1)+',
)

def mask_text(text: str,
              named_rules: list[tuple[re.Pattern, str]]) -> str:
    """
    Apply all masking layers to a single text string.
    Order: structural regex → named-entity blocklist.
    spaCy NER is applied separately in batch after this pass.
    """
    if not text:
        return text

    # Layer 1: structural patterns
    for pat, ph in STRUCTURAL_MASKS:
        text = pat.sub(f' {ph} ', text)

    # Layer 2: named-entity blocklist (platforms + clients + employees)
    for pat, ph in named_rules:
        text = pat.sub(f' {ph} ', text)

    # Collapse repeated identical placeholders
    text = _REPEATED_PH.sub(r'\1', text)
    # Collapse whitespace
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ===========================================================================
# 6. spaCy NER residual pass (optional, applied in batch)
# ===========================================================================
_SPACY_TARGET_LABELS = {'PERSON', 'ORG'}

def _mask_spacy_batch(texts: list[str]) -> list[str]:
    """
    Run spaCy NER on a list of already-blocklist-masked texts.
    Replaces residual PERSON → [CLIENT], ORG (non-platform) → [CLIENT].
    Returns the same list with in-place replacement.
    """
    if not SPACY_AVAILABLE:
        return texts

    results = []
    for doc in _nlp.pipe(texts, batch_size=128, disable=['tagger', 'parser', 'lemmatizer']):
        text = doc.text
        spans = []
        for ent in doc.ents:
            if ent.label_ not in _SPACY_TARGET_LABELS:
                continue
            if '[' in ent.text:          # already replaced by blocklist
                continue
            if ent.text.lower() in _PLATFORM_LOWER:  # known platform
                continue
            if len(ent.text) < 4:        # too short to be a reliable name
                continue
            spans.append((ent.start_char, ent.end_char, PH["client"]))

        if spans:
            for start, end, ph in sorted(spans, key=lambda x: -x[0]):
                text = text[:start] + f' {ph} ' + text[end:]
            text = re.sub(r'[ \t]{2,}', ' ', text).strip()
        results.append(text)
    return results


# ===========================================================================
# 7. Notes cleaning (same as preprocess.py)
# ===========================================================================
_BOILERPLATE_PATTERNS = [
    re.compile(r'-{3,}\s*\nThis task was submitted through TOS.*', re.DOTALL | re.IGNORECASE),
    re.compile(r'https?://\S+'),
    re.compile(r'app\.asana\.com/\S*'),
    re.compile(
        r'(?:best regards|kind regards|regards,|thank you in advance)[,.\s\n].*',
        re.DOTALL | re.IGNORECASE),
    re.compile(r'\[insert [^\]]+\]', re.IGNORECASE),
    re.compile(r'\*{2,}'),
    re.compile(r'^\s*-{4,}\s*$', re.MULTILINE),
]

def clean_notes(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    for pat in _BOILERPLATE_PATTERNS:
        text = pat.sub(' ', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ===========================================================================
# 8. Form field extraction (same as preprocess.py)
# ===========================================================================
_NEXT_FIELD_LA = r'(?=\n[A-Z][^\n:]{2,80}[?]?:|\Z)'

def extract_field(notes_text: str, label: str) -> str:
    if not notes_text:
        return ""
    pat = re.compile(
        re.escape(label) + r'\s*\n(.*?)' + _NEXT_FIELD_LA,
        re.DOTALL | re.IGNORECASE,
    )
    m = pat.search(notes_text)
    return m.group(1).strip() if m else ""


# ===========================================================================
# MAIN
# ===========================================================================
if not INPUT_FILE.exists():
    sys.exit(f"ERROR: '{INPUT_FILE}' not found. Place it in the same folder.")

print(f"Loading {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False)
df.columns = df.columns.str.strip()
df = df.map(lambda s: "" if pd.isna(s) else str(s).strip())
n_raw = len(df)
print(f"  {n_raw:,} rows, {df.shape[1]} columns loaded")

# ---------------------------------------------------------------------------
# Filter chain (identical to preprocess.py)
# ---------------------------------------------------------------------------
print("\n[FILTER CHAIN]")
mask = df["Parent task"] != ""
df   = df[~mask].copy()
print(f"  Step 1  Drop sub-tasks          : -{int(mask.sum()):,} rows  (remaining: {len(df):,})")

mask = df["Name"].str.contains("KaizenBot", case=False, regex=False)
df   = df[~mask].copy()
print(f"  Step 2  Drop KaizenBot          : -{int(mask.sum()):,} rows  (remaining: {len(df):,})")

mask = df["Name"].str.contains("Working Session", case=False, regex=False)
df   = df[~mask].copy()
print(f"  Step 3  Drop Working Sessions   : -{int(mask.sum()):,} rows  (remaining: {len(df):,})")

mask = (df["Name"] == "") & (df["Notes"] == "")
df   = df[~mask].copy()
print(f"  Step 4  Drop fully empty rows   : -{int(mask.sum()):,} rows  (remaining: {len(df):,})")

# ---------------------------------------------------------------------------
# Form field extraction
# ---------------------------------------------------------------------------
print("\n[FORM FIELD EXTRACTION]")
df["form_intent"]     = df["Notes"].apply(lambda t: extract_field(t, "How may we assist you?:"))
df["form_intent_alt"] = df["Notes"].apply(lambda t: extract_field(t, "Please select the issue.:"))
df["form_intent"]     = df["form_intent"].where(df["form_intent"] != "", df["form_intent_alt"])
df.drop(columns=["form_intent_alt"], inplace=True)
df["form_detail"] = df["Notes"].apply(lambda t: extract_field(t, "Please specify.:"))

n_intent = int((df["form_intent"] != "").sum())
n_detail = int((df["form_detail"] != "").sum())
print(f"  form_intent extracted : {n_intent:,} rows ({n_intent/len(df)*100:.1f}%)")
print(f"  form_detail extracted : {n_detail:,} rows ({n_detail/len(df)*100:.1f}%)")

# ---------------------------------------------------------------------------
# Notes cleaning
# ---------------------------------------------------------------------------
print("\n[NOTES CLEANING]")
df["notes_clean"] = df["Notes"].apply(clean_notes)
notes_filled = int((df["notes_clean"] != "").sum())
print(f"  Rows with non-empty notes_clean : {notes_filled:,} ({notes_filled/len(df)*100:.1f}%)")

# ---------------------------------------------------------------------------
# Build entity blocklist
# ---------------------------------------------------------------------------
print("\n[BUILDING ENTITY BLOCKLISTS]")

# -- Platforms (typed categories) --------------------------------------------
# Each category maps to its own semantic placeholder so BERTopic sees
# "[PORTFOLIO_PLATFORM]" vs "[CUSTODIAN]" vs "[BANK]" as distinct signals.
platform_pairs: list[tuple[str, str]] = []
n_platform_total = 0
for placeholder, terms in PLATFORM_CATEGORIES.items():
    for term in terms:
        for v in _make_variants(term):
            platform_pairs.append((v, placeholder))
    n_platform_total += len(terms)

n_platform_variants = len({v for v, _ in platform_pairs})
n_categories = len(PLATFORM_CATEGORIES)
print(f"  Platform categories    : {n_categories:>3} types, {n_platform_total:>3} terms -> {n_platform_variants:>3} variants")
for ph, terms in PLATFORM_CATEGORIES.items():
    print(f"    {ph:<28}: {len(terms)} terms")

# -- Clients / Households (dynamic from Household column) --------------------
client_names: set[str] = set()
SKIP_HH = {'n/a', 'firm-wide', '', 'firm wide'}
if "Household" in df.columns:
    for cell in df["Household"].unique():
        for part in cell.split(','):
            part = part.strip()
            if part.lower() not in SKIP_HH and part:
                client_names.add(part)

client_pairs: list[tuple[str, str]] = []
for name in client_names:
    for v in _make_variants(name):
        # Skip very generic words that appear in platform list or are common financial terms
        if v.lower() not in {'capital', 'creek', 'venture', 'origin', 'mighty', 'gary',
                              'ken', 'dean', 'rob', 'alex', 'mark', 'elaine', 'russ',
                              'lynley', 'caren', 'brian', 'carl', 'susan', 'sheryl',
                              'larry', 'woody', 'bald', 'tim', 'sam'}:
            client_pairs.append((v, PH["client"]))

n_client_variants = len({v for v, _ in client_pairs})
print(f"  Client/household names : {len(client_names):>3} entries -> {n_client_variants:>3} variants")

# -- Employees (from Assignee column) ----------------------------------------
employee_names: set[str] = set()
if "Assignee" in df.columns:
    for cell in df["Assignee"].unique():
        cell = cell.strip()
        if cell and '@' not in cell and cell:
            employee_names.add(cell)
        elif '@' in cell:
            # email-based assignee: extract name part
            local = cell.split('@')[0]
            # kavin.pandey -> Kavin Pandey
            parts = local.replace('.', ' ').replace('_', ' ').title()
            employee_names.add(parts)

# Also add first-name-only variants for employees (they appear alone in task names)
# e.g. "Rodrigo Besoy - Addepar updates" -> "Rodrigo" is not in employee list but is a name
# spaCy handles these; for the blocklist we add full names only
employee_pairs: list[tuple[str, str]] = []
for name in employee_names:
    for v in _make_variants(name):
        employee_pairs.append((v, PH["employee"]))

n_emp_variants = len({v for v, _ in employee_pairs})
print(f"  Employee names         : {len(employee_names):>3} entries -> {n_emp_variants:>3} variants")

# Compile all rules (platform rules applied first to prevent CLIENT overriding PLATFORM)
all_rules = build_rules(platform_pairs + client_pairs + employee_pairs)
print(f"  Total compiled rules   : {len(all_rules):>3}")

# ---------------------------------------------------------------------------
# Apply masking to Name and notes_clean
# ---------------------------------------------------------------------------
print("\n[APPLYING ENTITY MASKS]")
print("  Masking Name column ...")
df["name_masked"]  = df["Name"].apply(lambda t: mask_text(t, all_rules))

print("  Masking form_intent column (platform names in dropdown values) ...")
df["form_intent_masked"] = df["form_intent"].apply(lambda t: mask_text(t, all_rules))

print("  Masking notes_clean column ...")
df["notes_masked"] = df["notes_clean"].apply(lambda t: mask_text(t, all_rules))

# spaCy residual pass (batch for efficiency)
if SPACY_AVAILABLE:
    print("  Running spaCy NER residual pass on Name ...")
    df["name_masked"]  = _mask_spacy_batch(df["name_masked"].tolist())
    print("  Running spaCy NER residual pass on notes_masked ...")
    df["notes_masked"] = _mask_spacy_batch(df["notes_masked"].tolist())

# ---------------------------------------------------------------------------
# Build complaint_text (raw) and complaint_text (masked)
# ---------------------------------------------------------------------------
print("\n[BUILDING complaint_text]")

def _build_text(row, intent_col, name_col, notes_col) -> str:
    parts = []
    if row[intent_col]:
        parts.append(row[intent_col])
    parts.append(row[name_col])
    if row[notes_col]:
        parts.append(row[notes_col])
    return " ".join(p for p in parts if p).strip()

# Raw (unmasked) — kept for comparison and audit
df["complaint_text_raw"] = df.apply(
    lambda r: _build_text(r, "form_intent", "Name", "notes_clean"), axis=1)

# Masked — this is what goes into BERTopic
# form_intent_masked used here so "Addepar updates" -> "[PLATFORM] updates"
df["complaint_text"] = df.apply(
    lambda r: _build_text(r, "form_intent_masked", "name_masked", "notes_masked"), axis=1)

# Drop rows whose masked complaint_text is too short
before = len(df)
df = df[df["complaint_text"].str.len() >= MIN_TEXT_LEN].copy()
dropped = before - len(df)
if dropped:
    print(f"  Dropped {dropped:,} rows where complaint_text < {MIN_TEXT_LEN} chars after masking")

# ---------------------------------------------------------------------------
# Deduplicate identical complaint_text values
# ---------------------------------------------------------------------------
# Recurring tasks (e.g. "Check all [PORTFOLIO_PLATFORM] capital calls" ×523)
# produce hundreds of identical embeddings. HDBSCAN treats them as an
# artificially dense cloud and micro-fragments them across multiple topics.
# Solution: keep one representative row per unique text; store the repetition
# count in 'dup_count' so downstream analysis can weight by frequency.
#
# Deduplication is on the MASKED text (complaint_text) — two tasks with
# different client names but the same operation collapse correctly.
# Raw columns (Name, Notes, complaint_text_raw) retain the first occurrence.
print("\n[DEDUPLICATION]")
before_dedup = len(df)
df["dup_count"] = df.groupby("complaint_text")["complaint_text"].transform("count")
df = df.drop_duplicates(subset="complaint_text", keep="first").copy()
df = df.reset_index(drop=True)
after_dedup = len(df)
removed = before_dedup - after_dedup
dup_docs = df[df["dup_count"] > 1]
print(f"  Before dedup : {before_dedup:,} rows")
print(f"  After  dedup : {after_dedup:,} rows  (-{removed:,} removed)")
print(f"  Unique texts : {after_dedup:,}  |  Tasks with repetitions: {len(dup_docs):,}")
print(f"  Top 10 most repeated task patterns:")
top_dups = df.nlargest(10, "dup_count")[["Name", "dup_count"]]
for _, r in top_dups.iterrows():
    print(f"    {r['dup_count']:>4}x  {r['Name'][:75]}")

# ---------------------------------------------------------------------------
# Masking impact stats
# ---------------------------------------------------------------------------
print("\n[MASKING IMPACT STATS]")
import re as _re

def count_placeholder(col, ph):
    return int(df[col].str.count(re.escape(ph)).sum())

print("  Platform category placeholders:")
for placeholder in PLATFORM_CATEGORIES:
    n = count_placeholder("complaint_text", placeholder)
    if n > 0:
        print(f"    {placeholder:<28}: {n:,}")

print("  Entity placeholders:")
for label in ("client", "employee", "email", "entity_id", "phone", "url"):
    ph = PH[label]
    n = count_placeholder("complaint_text", ph)
    print(f"    {ph:<28}: {n:,}")

# Before/after vocabulary comparison
raw_tokens   = _re.findall(r'[A-Za-z]{4,}', ' '.join(df["complaint_text_raw"].tolist()))
masked_tokens = _re.findall(r'[A-Za-z]{4,}', ' '.join(df["complaint_text"].tolist()))
from collections import Counter
raw_top5    = Counter(t.lower() for t in raw_tokens).most_common(10)
masked_top5 = Counter(t.lower() for t in masked_tokens
                      if not t.startswith('[') and t.lower() not in
                      {'please', 'that', 'this', 'with', 'from', 'have', 'been',
                       'will', 'they', 'their', 'into', 'task', 'name', 'email'}).most_common(10)
print("\n  Top 10 tokens BEFORE masking:")
for tok, cnt in raw_top5:
    print(f"    {tok:<20} {cnt:,}")
print("\n  Top 10 tokens AFTER masking (excl. stopwords):")
for tok, cnt in masked_top5:
    print(f"    {tok:<20} {cnt:,}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
KEEP_COLS = [
    "Task ID", "Created At", "Completed At", "Name", "Section/Column",
    "Assignee", "Household", "Priority", "Issue type", "Issue sub-type",
    "Issue category", "Tags", "Task Progress", "Notes",
    "form_intent", "form_intent_masked", "form_detail",
    "notes_clean", "notes_masked",
    "complaint_text_raw", "complaint_text", "dup_count",
]
out_cols = [c for c in KEEP_COLS if c in df.columns]
df[out_cols].to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PREPROCESSING v2 COMPLETE")
print("=" * 60)
print(f"  Raw input rows         : {n_raw:,}")
print(f"  Rows removed           : {n_raw - len(df):,}")
print(f"  Final corpus size      : {len(df):,}")
print()
print(f"  Rows with form_intent  : {int((df['form_intent']!='').sum()):,} "
      f"({(df['form_intent']!='').mean()*100:.1f}%)")
print(f"  Rows with notes_masked : {int((df['notes_masked']!='').sum()):,} "
      f"({(df['notes_masked']!='').mean()*100:.1f}%)")
print()

# Show 3 before/after examples
print("  SAMPLE BEFORE/AFTER (3 random rows):")
sample = df.sample(3, random_state=42)
for _, row in sample.iterrows():
    print(f"\n  RAW   : {row['complaint_text_raw'][:200]}")
    print(f"  MASKED: {row['complaint_text'][:200]}")

print()
print(f"  Saved -> {OUTPUT_FILE}")
print(f"  New columns: notes_masked, complaint_text_raw")
print("=" * 60)
print()
print("NEXT STEP: re-run model_pipeline.py to retrain BERTopic on anonymized text.")