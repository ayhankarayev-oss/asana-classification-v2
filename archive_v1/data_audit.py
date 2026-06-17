"""
Phase 0 - Data Audit
====================
Run : python data_audit.py
Goal: Understand the raw shape of Client_support.csv before any modelling.
      Paste the full console output back to Claude for review.
"""

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
CSV_FILE   = Path("Client_support.csv")
SAMPLE_N   = 20      # rows shown in the random-sample section
TOP_N_TAGS = 30      # max unique values shown per tag column

W    = 72
SEP  = "=" * W
SEP2 = "-" * W

def section(title):
    print("\n" + SEP + "\n  " + title + "\n" + SEP)

def clean(s):
    """Return stripped string; treat NaN / None / whitespace-only as empty."""
    if pd.isna(s):
        return ""
    return str(s).strip()

# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
if not CSV_FILE.exists():
    sys.exit(
        "ERROR: '{}' not found. "
        "Place it in the same folder as this script and re-run.".format(CSV_FILE)
    )

print("\nLoading {} ...".format(CSV_FILE))
df = pd.read_csv(
    CSV_FILE,
    dtype=str,             # keep everything as text - no silent type coercions
    keep_default_na=False  # empty cells -> "" not NaN
)
df.columns = df.columns.str.strip()

# pandas renames duplicate column names automatically:
# col [04] "Name"   = Asana task name  (what we classify on)
# col [51] "Name.1" = investor/contact name field (mostly empty)
TASK_NAME_COL = "Name"

# Replace every cell with a clean string (empty string when blank/NaN)
df = df.map(clean)

total = len(df)

# --------------------------------------------------------------------------
# SECTION 1 - Shape & columns
# --------------------------------------------------------------------------
section("1. SHAPE & COLUMNS")
print("  Rows    : {:,}".format(total))
print("  Columns : {}".format(df.shape[1]))
print("\n  All column names:")
for i, col in enumerate(df.columns):
    marker = " << duplicate 'Name' column" if col == "Name.1" else ""
    print("    [{:02d}] {}{}".format(i, col, marker))

# --------------------------------------------------------------------------
# SECTION 2 - Key column emptiness
# --------------------------------------------------------------------------
section("2. KEY COLUMN EMPTINESS")

KEY_COLS = [
    "Name", "Notes", "Issue type", "Issue sub-type",
    "Issue category", "Tags", "Section/Column", "Parent task",
    "Task Progress", "Priority", "Household",
]
print("  {:<30} {:>7}  {:>7}  {:>9}".format("Column", "Filled", "Empty", "% Filled"))
print("  " + SEP2)
for col in KEY_COLS:
    if col not in df.columns:
        print("  {:<30} ** COLUMN NOT FOUND **".format(col))
        continue
    filled = (df[col] != "").sum()
    empty  = total - filled
    pct    = filled / total * 100
    print("  {:<30} {:>7,}  {:>7,}  {:>8.1f}%".format(col, filled, empty, pct))

# --------------------------------------------------------------------------
# SECTION 3 - Sub-task vs top-level split
# --------------------------------------------------------------------------
section("3. SUB-TASKS vs TOP-LEVEL TASKS")

is_subtask = df["Parent task"] != ""
n_subtasks = int(is_subtask.sum())
n_top      = total - n_subtasks
print("  Top-level tasks : {:,}  ({:.1f}%)".format(n_top,      n_top / total * 100))
print("  Sub-tasks       : {:,}  ({:.1f}%)".format(n_subtasks, n_subtasks / total * 100))
print("\n  Sub-tasks have a Parent task value and are workflow checklist")
print("  steps - they should be excluded from the classification model.")

# --------------------------------------------------------------------------
# SECTION 4 - Automated / system task detection
# --------------------------------------------------------------------------
section("4. AUTOMATED & SYSTEM TASKS")

kaizen_mask   = df[TASK_NAME_COL].str.contains("KaizenBot",         case=False, regex=False)
onboard_mask  = df[TASK_NAME_COL].str.contains("[Onboarding]",      case=False, regex=False)
worksess_mask = df[TASK_NAME_COL].str.contains("Working Session",   case=False, regex=False)
formsubm_mask = df["Notes"].str.contains("submitted through TOS",   case=False, regex=False)

print("  {:<40} {:>6}  {}".format("Type", "Count", "Note"))
print("  " + SEP2)
print("  {:<40} {:>6,}  {}".format("KaizenBot alerts",                  kaizen_mask.sum(),   "Auto data-quality flags -> own category"))
print("  {:<40} {:>6,}  {}".format("[Onboarding] tasks",                onboard_mask.sum(),  "Onboarding workflow tasks"))
print("  {:<40} {:>6,}  {}".format("Working Session records",           worksess_mask.sum(), "Internal meetings, no complaint text"))
print("  {:<40} {:>6,}  {}".format("Form-submitted (TOS form)",         formsubm_mask.sum(), "Via Asana form - richest text signal"))
print("\n  Note: rows may match more than one pattern above.")

# --------------------------------------------------------------------------
# SECTION 5 - Existing classification tag distributions
# --------------------------------------------------------------------------
TAG_COLS = [
    "Issue type", "Issue sub-type", "Issue category",
    "Section/Column", "Tags", "Priority", "Task Progress",
]

for col in TAG_COLS:
    if col not in df.columns:
        continue
    non_empty = df[df[col] != ""][col]
    if non_empty.empty:
        continue

    section("5. DISTRIBUTION - '{}'  (non-empty: {:,} / {:,})".format(col, len(non_empty), total))
    vc        = non_empty.value_counts()
    max_count = vc.iloc[0]
    for label, count in vc.head(TOP_N_TAGS).items():
        bar = "#" * max(1, int(count / max_count * 38))
        print("  {:>6,}  {:<38}  {}".format(count, bar, label))
    if len(vc) > TOP_N_TAGS:
        print("  ... and {} more unique values".format(len(vc) - TOP_N_TAGS))

# --------------------------------------------------------------------------
# SECTION 6 - Text length distributions
# --------------------------------------------------------------------------
section("6. TEXT LENGTH DISTRIBUTIONS")

BUCKETS = [
    (0,    0,    "  0 chars (empty) "),
    (1,    50,   "  1-50 chars      "),
    (51,   150,  " 51-150 chars     "),
    (151,  300,  "151-300 chars     "),
    (301,  600,  "301-600 chars     "),
    (601,  1200, "601-1200 chars    "),
    (1201, 10**9,"   >1200 chars    "),
]

for col_name in ["Name", "Notes"]:
    if col_name not in df.columns:
        continue
    lengths  = df[col_name].str.len()
    non_zero = lengths[lengths > 0]
    print("\n  -- {} --".format(col_name))
    print("  Empty rows   : {:,}".format(int((lengths == 0).sum())))
    if len(non_zero):
        print("  Min          : {:,} chars".format(int(non_zero.min())))
        print("  Median       : {:,} chars".format(int(non_zero.median())))
        print("  Mean         : {:.1f} chars".format(float(non_zero.mean())))
        print("  Max          : {:,} chars".format(int(non_zero.max())))
        print()
        for lo, hi, label in BUCKETS:
            n   = int((lengths == 0).sum()) if lo == 0 else int(((lengths >= lo) & (lengths <= hi)).sum())
            bar = "#" * max(0, int(n / total * 40))
            print("  {} : {:>6,}  {}".format(label, n, bar))

# --------------------------------------------------------------------------
# SECTION 7 - Form field structure inside Notes
# --------------------------------------------------------------------------
section("7. STRUCTURED FORM FIELDS DETECTED IN Notes")

FORM_LABEL_RE = re.compile(r"^([A-Z][^\n:]{3,80}[?]?):", re.MULTILINE)

all_labels      = []
notes_with_form = 0

for note in df["Notes"]:
    matches = FORM_LABEL_RE.findall(note)
    if matches:
        notes_with_form += 1
        all_labels.extend(m.strip() for m in matches)

label_counts = Counter(all_labels)
print("  Notes with form-style fields : {:,}".format(notes_with_form))
print("\n  Top 20 form field labels (by occurrence across all records):")
print("  {:>6}  {}".format("Count", "Label"))
print("  " + SEP2)
for label, count in label_counts.most_common(20):
    print("  {:>6,}  {}".format(count, label))

# --------------------------------------------------------------------------
# SECTION 8 - Rows where both Name AND Notes are empty
# --------------------------------------------------------------------------
section("8. ROWS WITH BOTH Name AND Notes EMPTY")

both_empty = int(((df["Name"] == "") & (df["Notes"] == "")).sum())
print("  Count : {:,}".format(both_empty))
if both_empty:
    print("  These rows carry no text signal and will be dropped before modelling.")
else:
    print("  None - every row has at least some text.")

# --------------------------------------------------------------------------
# SECTION 9 - Unique value counts for classification columns
# --------------------------------------------------------------------------
section("9. UNIQUE VALUE COUNTS - CLASSIFICATION COLUMNS")

print("  {:<28} {:>25}".format("Column", "Unique non-empty values"))
print("  " + SEP2)
for col in ["Issue type", "Issue sub-type", "Issue category", "Tags"]:
    if col in df.columns:
        n = df[df[col] != ""][col].nunique()
        print("  {:<28} {:>25,}".format(col, n))

# --------------------------------------------------------------------------
# SECTION 10 - Random sample: Name + Notes preview
# --------------------------------------------------------------------------
section("10. RANDOM SAMPLE - {} ROWS WITH NON-EMPTY Notes".format(SAMPLE_N))

has_notes = df[df["Notes"] != ""]
sample    = has_notes.sample(min(SAMPLE_N, len(has_notes)), random_state=42)

for i, (_, row) in enumerate(sample.iterrows(), 1):
    name_str  = row["Name"][:100]
    issue_str = "{} / {}".format(row.get("Issue type", ""), row.get("Issue sub-type", "")).strip(" /")
    notes_str = row["Notes"].replace("\n", " | ")[:220]
    print("\n  [{:02d}]".format(i))
    print("  NAME  : {}".format(name_str))
    print("  ISSUE : {}".format(issue_str if issue_str else "(none)"))
    print("  NOTES : {}".format(notes_str))
    print("  " + "-" * 68)

# --------------------------------------------------------------------------
# SECTION 11 - Estimated classifiable corpus size
# --------------------------------------------------------------------------
section("11. ESTIMATED CLASSIFIABLE CORPUS SIZE")

top_level_df = df[~is_subtask]
no_kaizen    = top_level_df[~top_level_df[TASK_NAME_COL].str.contains("KaizenBot",      case=False, regex=False)]
no_worksess  = no_kaizen[~no_kaizen[TASK_NAME_COL].str.contains("Working Session",      case=False, regex=False)]
has_any_text = no_worksess[(no_worksess["Name"] != "") | (no_worksess["Notes"] != "")]

minus_kaizen    = int(kaizen_mask[~is_subtask].sum())
minus_worksess  = int(worksess_mask[~is_subtask & ~kaizen_mask].sum())
minus_notext    = int((~is_subtask & ~kaizen_mask & ~worksess_mask & (df["Name"] == "") & (df["Notes"] == "")).sum())

print("  Total records                       : {:,}".format(total))
print("  Minus sub-tasks                     : -{:,}".format(n_subtasks))
print("  Minus KaizenBot alerts              : -{:,}".format(minus_kaizen))
print("  Minus Working Sessions              : -{:,}".format(minus_worksess))
print("  Minus rows with no text at all      : -{:,}".format(minus_notext))
print("  " + SEP2)
print("  Estimated classifiable corpus       :  {:,} rows".format(len(has_any_text)))
print("\n  Of those, rows WITH Notes           :  {:,}".format(int((has_any_text["Notes"] != "").sum())))
print("  Of those, rows with ONLY Name text  :  {:,}".format(int((has_any_text["Notes"] == "").sum())))

# --------------------------------------------------------------------------
print("\n" + SEP)
print("  AUDIT COMPLETE")
print("  Paste everything above back into the chat.")
print(SEP + "\n")