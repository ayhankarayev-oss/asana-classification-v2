"""
Central config for asana-classification-v2.
Edit values here — all pipeline steps import from this file.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
OUTPUT_DIR  = Path("outputs")

INPUT_CSV          = DATA_DIR   / "Client_support.csv"
CORPUS_CLEAN_CSV   = OUTPUT_DIR / "corpus_clean.csv"
CLASSIFIED_CSV     = OUTPUT_DIR / "complaints_classified.csv"
TOPIC_EXPLAIN_CSV  = OUTPUT_DIR / "topic_explanations.csv"
CLUSTER_REPORT_MD  = OUTPUT_DIR / "cluster_summary_report.md"
CLASSIFIER_PKL     = OUTPUT_DIR / "issue_classifier.pkl"
CLASSIFIER_REPORT  = OUTPUT_DIR / "classifier_report.txt"

# ── Preprocessing (preprocess.py) ─────────────────────────────────────────────
MIN_TEXT_LEN   = 15       # v2 increase: 10→15 — drop single-word tasks
SPACY_MODEL    = "en_core_web_lg"   # v2: upgrade md→lg for better NER recall
SPACY_FALLBACK = "en_core_web_sm"

# ── Embedding (shared by BERTopic + Classifier) ───────────────────────────────
# v2 change: MiniLM-L6 (384-dim) → mpnet-base (768-dim) — better semantic quality
# trade-off: ~2× slower on CPU, noticeably higher topic coherence on short texts
EMBEDDING_MODEL = "all-mpnet-base-v2"
RANDOM_STATE    = 42

# ── BERTopic (topic_model.py) ─────────────────────────────────────────────────
TARGET_TOPICS         = 20
MIN_CLUSTER_SIZE      = 40
TOP_N_WORDS           = 10
MMR_DIVERSITY         = 0.3
ZEROSHOT_MIN_SIMILARITY = 0.28    # v2: slightly lower threshold — reduces outlier rate

UMAP_N_NEIGHBORS  = 15    # v2: 20→15 — tighter local structure, fewer merged clusters
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST     = 0.0
UMAP_METRIC       = "cosine"

# Zero-shot seed topics — plain English descriptions of each operational category.
# v2 adds 4 new seeds that captured outlier patterns from the v1 validation report.
ZEROSHOT_TOPICS = [
    "new bank brokerage custodian account setup connection feed link online portal access",
    "account update maintenance close rename attribute change classification portfolio platform",
    "new private investment fund create entry position portfolio platform setup structure direct owner",
    "investment update valuation price change commitment amount capital unfunded transaction",
    "capital call distribution custodian payment processing transfer wire verify confirm receipt",
    "weekly periodic check all pending capital calls verify status mark complete audit portfolio platform",
    "missing data attribute cost basis quality incorrect error fix export reconcile",
    "statement document upload file retrieve download archive K1 invoice send billing document platform",
    "report PDF reporting portfolio platform view dashboard performance deliver generate quarterly",
    "ownership structure entity LLC trust beneficiary direct owner setup change legal",
    "recurring weekly monthly transaction reconciliation audit verify confirm status accounting platform",
    "onboarding new household banker introduction portal access setup client portal",
    "follow up reach out email response pending waiting reminder investigate confirm escalation",
    "portfolio platform view viewset create configure user access permissions data aggregator document",
    "general update ad hoc miscellaneous operational priority task specify",
    # v2 new seeds — derived from v1 outlier analysis (118 unclassified tasks)
    "tax preparation estimated taxes quarterly annual gathering data calculation",
    "payroll salary processing payment employee compensation",
    "vendor third party external coordinator mortgage real estate property",
    "client meeting sync call review discussion coordination",
]

# ── Classifier (classifier.py) ────────────────────────────────────────────────
# v2 change: TF-IDF features → sentence embedding features.
# Rationale: same embedding used by BERTopic gives the classifier richer semantic
# features, especially for short task names with few lexical cues.
CLASSIFIER_TEST_SIZE     = 0.20
CLASSIFIER_CV_FOLDS      = 5
# Confidence threshold: predictions below this are flagged for manual review
CONFIDENCE_THRESHOLD     = 0.60

# Human-readable business labels mapped to topic IDs.
# Update these after each BERTopic run by reading outputs/cluster_summary_report.md
BUSINESS_LABELS: dict[int, str] = {
    -1: "Outliers / Unclassified",
     0: "New Account & Data Feed Setup",
     1: "Portfolio Platform Account Updates",
     2: "New Private Investment Entry",
     3: "Private Investment Updates & Valuations",
     4: "Trust & Estate Cash Flow Verification",
     5: "Capital Call Audit & Monthly Statement Review",
     6: "Cost Basis & Data Quality Fixes",
     7: "Document Upload & Client Billing",
     8: "Reporting & Performance Analytics",
     9: "Ownership Structure & Legal Entity Changes",
    10: "Real Asset Transaction Audit",
    11: "Loan & Lending Account Setup",
    12: "Direct Deal Updates & Unlinked Accounts",
    13: "Portfolio Platform View & Access Configuration",
    14: "General & Ad Hoc Requests",
    15: "Tax Preparation & Estimated Taxes",
    16: "Payroll & Compensation Processing",
    17: "Third-party Vendor & Mortgage Coordination",
    18: "Client Meetings & Sync Calls",
}