"""
Phase 2 - Text Preprocessing
==============================
Reads : Client_support.csv
Writes: corpus_clean.csv

Steps
-----
1. Apply exact filter chain (sub-tasks, KaizenBot, Working Sessions, empty rows)
2. Extract structured form fields from Notes into dedicated columns
3. Strip HTML entities, URLs, boilerplate from Notes
4. Blend Name + Notes into a single 'complaint_text' column
"""

import re
import html
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
INPUT_FILE  = Path("Client_support.csv")
OUTPUT_FILE = Path("corpus_clean.csv")

MIN_TEXT_LEN = 10   # drop rows whose complaint_text is shorter than this

# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
if not INPUT_FILE.exists():
    sys.exit("ERROR: '{}' not found. Place it in the same folder.".format(INPUT_FILE))

print("Loading {} ...".format(INPUT_FILE))
df = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False)
df.columns = df.columns.str.strip()

# Normalise every cell: strip whitespace, convert NaN to empty string
df = df.map(lambda s: "" if pd.isna(s) else str(s).strip())
n_raw = len(df)
print("  {:,} rows, {} columns loaded".format(n_raw, df.shape[1]))

# --------------------------------------------------------------------------
# Filter Chain
# --------------------------------------------------------------------------
print("\n[FILTER CHAIN]")

# Step 1 - Drop sub-tasks (have a Parent task value)
mask = df["Parent task"] != ""
df   = df[~mask].copy()
print("  Step 1  Drop sub-tasks          : -{:,} rows  (remaining: {:,})".format(int(mask.sum()), len(df)))

# Step 2 - Drop KaizenBot auto-generated alerts
mask = df["Name"].str.contains("KaizenBot", case=False, regex=False)
df   = df[~mask].copy()
print("  Step 2  Drop KaizenBot          : -{:,} rows  (remaining: {:,})".format(int(mask.sum()), len(df)))

# Step 3 - Drop Working Session internal records
mask = df["Name"].str.contains("Working Session", case=False, regex=False)
df   = df[~mask].copy()
print("  Step 3  Drop Working Sessions   : -{:,} rows  (remaining: {:,})".format(int(mask.sum()), len(df)))

# Step 4 - Drop rows where both Name AND Notes are empty
mask = (df["Name"] == "") & (df["Notes"] == "")
df   = df[~mask].copy()
print("  Step 4  Drop fully empty rows   : -{:,} rows  (remaining: {:,})".format(int(mask.sum()), len(df)))

# --------------------------------------------------------------------------
# Form Field Extraction
# --------------------------------------------------------------------------
print("\n[FORM FIELD EXTRACTION]")

# Asana form structure in Notes looks like:
#
#   Field label:
#   <answer text>
#
#   Next field label:
#   <answer text>
#
# The lookahead stops capture at the next capitalised field label.
_NEXT_FIELD_LOOKAHEAD = r"(?=\n[A-Z][^\n:]{2,80}[?]?:|\Z)"

def extract_field(notes_text, label):
    """
    Pull the answer that follows 'label' in a structured Notes field block.
    Returns an empty string when the label is not found.
    """
    if not notes_text:
        return ""
    pattern = re.compile(
        re.escape(label) + r"\s*\n(.*?)" + _NEXT_FIELD_LOOKAHEAD,
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(notes_text)
    if match:
        return match.group(1).strip()
    return ""


# Primary intent label - present in the newer TOS intake form
df["form_intent"] = df["Notes"].apply(lambda t: extract_field(t, "How may we assist you?:"))

# Older form variant uses a different label for the same purpose
df["form_intent_alt"] = df["Notes"].apply(lambda t: extract_field(t, "Please select the issue.:"))

# Merge: use form_intent_alt only when form_intent is missing
df["form_intent"] = df["form_intent"].where(df["form_intent"] != "", df["form_intent_alt"])
df.drop(columns=["form_intent_alt"], inplace=True)

# Additional free-text detail supplied in the form
df["form_detail"] = df["Notes"].apply(lambda t: extract_field(t, "Please specify.:"))

n_intent = int((df["form_intent"] != "").sum())
n_detail = int((df["form_detail"] != "").sum())
print("  form_intent extracted : {:,} rows ({:.1f}%)".format(n_intent, n_intent / len(df) * 100))
print("  form_detail extracted : {:,} rows ({:.1f}%)".format(n_detail, n_detail / len(df) * 100))

# --------------------------------------------------------------------------
# Notes Cleaning
# --------------------------------------------------------------------------
print("\n[NOTES CLEANING]")

# Ordered list of regex patterns to strip from Notes before building complaint_text.
# Removing these reduces noise without losing the actual issue description.
STRIP_PATTERNS = [
    # Asana form submission footer (boilerplate, adds no classification value)
    re.compile(r"-{3,}\s*\nThis task was submitted through TOS.*", re.DOTALL | re.IGNORECASE),
    # Full URLs (links to Egnyte, Addepar, Asana - not useful for topic modeling)
    re.compile(r"https?://\S+"),
    re.compile(r"app\.asana\.com/\S*"),
    # Generic email sign-offs
    re.compile(r"(?:best regards|kind regards|regards,|thank you in advance)[,.\s\n].*",
               re.DOTALL | re.IGNORECASE),
    # Template placeholder text
    re.compile(r"\[insert [^\]]+\]", re.IGNORECASE),
    # Markdown bold markers (double asterisks)
    re.compile(r"\*{2,}"),
    # Lines that are only dashes (horizontal rules)
    re.compile(r"^\s*-{4,}\s*$", re.MULTILINE),
]


def clean_notes(text):
    if not text:
        return ""
    text = html.unescape(text)               # resolve &amp; &#39; etc.
    text = re.sub(r"<[^>]+>", " ", text)     # strip any residual HTML tags
    for pat in STRIP_PATTERNS:
        text = pat.sub(" ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)   # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse 3+ blank lines to 2
    return text.strip()


df["notes_clean"] = df["Notes"].apply(clean_notes)

notes_filled = int((df["notes_clean"] != "").sum())
print("  Rows with non-empty notes_clean : {:,} ({:.1f}%)".format(
    notes_filled, notes_filled / len(df) * 100))

# --------------------------------------------------------------------------
# Build complaint_text
# --------------------------------------------------------------------------
print("\n[BUILDING complaint_text]")

# Construction logic:
#
# 1. Lead with form_intent when available.
#    Reason: "New private investment" is the clearest possible classification
#    signal. Placing it first gives it disproportionate weight in the
#    sentence-transformer embedding without any manual weighting code.
#
# 2. Always include Name.
#    Reason: Present on 99.99% of rows. Even short titles like
#    "Ricardo Nazario, New loan" carry strong classification information.
#
# 3. Append notes_clean when non-empty.
#    Reason: Provides richer context for BERTopic's c-TF-IDF keyword
#    extraction and improves cluster separation.


def build_complaint_text(row):
    parts = []
    if row["form_intent"]:
        parts.append(row["form_intent"])
    parts.append(row["Name"])
    if row["notes_clean"]:
        parts.append(row["notes_clean"])
    return " ".join(p for p in parts if p).strip()


df["complaint_text"] = df.apply(build_complaint_text, axis=1)

# Drop anything that ended up shorter than MIN_TEXT_LEN after cleaning
before = len(df)
df = df[df["complaint_text"].str.len() >= MIN_TEXT_LEN].copy()
dropped = before - len(df)
if dropped:
    print("  Dropped {:,} rows where complaint_text < {} chars after cleaning".format(
        dropped, MIN_TEXT_LEN))

df = df.reset_index(drop=True)

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
KEEP_COLS = [
    "Task ID", "Created At", "Completed At", "Name", "Section/Column",
    "Assignee", "Household", "Priority", "Issue type", "Issue sub-type",
    "Issue category", "Tags", "Task Progress", "Notes",
    "form_intent", "form_detail", "notes_clean", "complaint_text",
]
out_cols = [c for c in KEEP_COLS if c in df.columns]
df[out_cols].to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
print("  Raw input rows         : {:,}".format(n_raw))
print("  Rows removed           : {:,}".format(n_raw - len(df)))
print("  Final corpus size      : {:,}".format(len(df)))
print("")
print("  Rows with form_intent  : {:,} ({:.1f}%)".format(
    int((df["form_intent"] != "").sum()),
    (df["form_intent"] != "").sum() / len(df) * 100))
print("  Rows with notes_clean  : {:,} ({:.1f}%)".format(
    int((df["notes_clean"] != "").sum()),
    (df["notes_clean"] != "").sum() / len(df) * 100))
print("  Name-only rows         : {:,} ({:.1f}%)".format(
    int((df["notes_clean"] == "").sum()),
    (df["notes_clean"] == "").sum() / len(df) * 100))
print("")
print("  complaint_text length  :")
ct_len = df["complaint_text"].str.len()
print("    Min    : {:,} chars".format(int(ct_len.min())))
print("    Median : {:,} chars".format(int(ct_len.median())))
print("    Mean   : {:.0f} chars".format(float(ct_len.mean())))
print("    Max    : {:,} chars".format(int(ct_len.max())))
print("")
print("  Saved -> {}".format(OUTPUT_FILE))
print("=" * 60)