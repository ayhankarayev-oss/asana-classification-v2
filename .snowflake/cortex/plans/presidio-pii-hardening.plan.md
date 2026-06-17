# Plan: Presidio PII Hardening + Text Normalization

## Problem Statement

**Current Issue:** Person names (Mike Albanese, Carolina Letsos, Harold Talisman) appear in `cleaned_text` because they're not in the Household/Assignee columns that the current PII masking targets.

**Root Cause:** 
- Current PII logic only masks names from DataFrame **columns** (Household, Assignee)
- Names appearing in the **text body itself** (task Name field, notes, descriptions) are not caught
- Case-sensitivity creates leakage risk (Addepar vs ADDEPAR vs addepar)

**User Requirement:**
1. Use Microsoft Presidio (ML-based) for comprehensive PII detection
2. Normalize `cleaned_text` to lowercase/standardized format
3. Eliminate ALL name/entity leakage

---

## Solution Architecture

### Three-Layer Defense

```
Text Input
    ↓
[1] Regex-based PII     → Email, phone, SSN patterns
    ↓
[2] Presidio ML Scan    → PERSON, ORGANIZATION, LOCATION (NEW)
    ↓
[3] Entity Extraction   → Platform/custodian/bank removal
    ↓
[4] Text Normalization  → Lowercase + decapitalization (NEW)
    ↓
cleaned_text (bias-free, normalized)
```

### Why This Order?

1. **Regex first**: Fast, deterministic patterns (email, phone)
2. **Presidio second**: ML-based entity detection catches names in freeform text
3. **Entity extraction third**: Domain-specific business entities (platforms, custodians)
4. **Normalization last**: Final pass to eliminate case-variant leakage

---

## Implementation Details

### Task 1: Install Presidio

```bash
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg
```

**Why en_core_web_lg?** Large model (780MB) has better accuracy for person/org names than small model.

---

### Task 2: Create Presidio Scanner Module

**File:** `pipeline/presidio_scanner.py`

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from typing import Optional

class PresidioScanner:
    """ML-based PII detection using Microsoft Presidio."""
    
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # Entity types to detect
        self.entities = [
            "PERSON",           # Mike Albanese, Carolina Letsos
            "EMAIL_ADDRESS",    # Backup for emails
            "PHONE_NUMBER",     # Backup for phones
            "ORGANIZATION",     # Company names (Mosaic Advisors, TOS)
            "LOCATION",         # Houston, TX, Oklahoma (optional)
            "US_SSN",          # SSN backup
            "CREDIT_CARD",     # Credit card backup
        ]
    
    def scan_and_mask(self, text: str, mask_location: bool = False) -> tuple[str, list]:
        """
        Scan text for PII and return masked text + detection log.
        
        Args:
            text: Input text
            mask_location: If True, mask location entities (may be too aggressive)
        
        Returns:
            (masked_text, detections_list)
        """
        if not text or not text.strip():
            return text, []
        
        # Analyze
        entities_to_detect = self.entities if mask_location else [
            e for e in self.entities if e != "LOCATION"
        ]
        
        results = self.analyzer.analyze(
            text=text,
            entities=entities_to_detect,
            language="en"
        )
        
        # Build detection log
        detections = []
        for result in results:
            detections.append({
                "entity_type": result.entity_type,
                "start": result.start,
                "end": result.end,
                "score": result.score,
                "text": text[result.start:result.end]
            })
        
        # Anonymize
        if results:
            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators={
                    "PERSON": {"type": "replace", "new_value": "[PERSON]"},
                    "ORGANIZATION": {"type": "replace", "new_value": "[ORG]"},
                    "EMAIL_ADDRESS": {"type": "replace", "new_value": "[EMAIL]"},
                    "PHONE_NUMBER": {"type": "replace", "new_value": "[PHONE]"},
                    "LOCATION": {"type": "replace", "new_value": "[LOCATION]"},
                    "US_SSN": {"type": "replace", "new_value": "[ID]"},
                    "CREDIT_CARD": {"type": "replace", "new_value": "[CREDIT_CARD]"},
                }
            )
            return anonymized.text, detections
        
        return text, []
```

**Why not mask LOCATION?** Locations like "Houston, TX" may be useful operational context. Make it configurable.

---

### Task 3: Integrate Presidio into preprocess.py

**Current flow:**
```python
# Step 5: Apply PII masking
text_pii = apply_pii(text_clean, pii_rules)

# Step 6: Extract entities + clean
result = preprocessor.process(text_pii, task_id=task_id)
```

**New flow:**
```python
# Step 5: Apply regex-based PII masking
text_pii = apply_pii(text_clean, pii_rules)

# Step 5b: Apply Presidio ML-based PII detection (NEW)
presidio = PresidioScanner()
text_presidio, presidio_log = presidio.scan_and_mask(text_pii, mask_location=False)
presidio_detections.extend(presidio_log)  # Track for audit

# Step 6: Extract entities + clean
result = preprocessor.process(text_presidio, task_id=task_id)
```

**Output:** New `outputs/presidio_detections.csv` with columns:
- task_id
- entity_type
- detected_text
- start_pos
- end_pos
- confidence_score

---

### Task 4: Text Normalization

**Function:** `normalize_text(text: str) -> str`

**Approach: Aggressive Decapitalization**

```python
def normalize_text(text: str) -> str:
    """
    Normalize text to prevent case-variant leakage.
    Strategy: Full lowercase to eliminate proper noun patterns.
    """
    if not text:
        return text
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove any remaining title-case artifacts (backup safety)
    # Example: "Mike Albanese" → "mike albanese" (already done by lower())
    
    return text
```

**Why full lowercase?**
- Embeddings (sentence-transformers) are case-insensitive by default
- BERTopic clustering benefits from normalized text
- Eliminates "Addepar" vs "ADDEPAR" vs "addepar" as distinct tokens
- Simple, deterministic, no edge cases

**Tradeoff:**
- ✅ Pro: Zero case-variant leakage, better clustering
- ⚠️ Con: Loses proper noun distinction (but we're masking those anyway)

---

### Task 5: Update Entity Extractor

**Current:** Already case-insensitive via `re.IGNORECASE` flag

**Change needed:** Ensure canonical output is still proper case (Addepar, not addepar)

```python
def _assign_platforms(self, entities: list[dict]) -> tuple[str, Optional[str]]:
    """Assign primary/secondary platform from extracted entities."""
    canonical = {"addepar": "Addepar", "arch": "Arch", ...}  # Already exists
    tracked_found = []
    for e in entities:
        key = e["entity"].lower()  # Now input is already lowercase
        if key in TRACKED_PLATFORMS:
            tracked_found.append(canonical[key])
    # ... rest unchanged
```

**No breaking changes** - extractor already handles case-insensitive matching.

---

### Task 6: Add Tests

**New tests in `tests/test_pipeline.py`:**

```python
def test_presidio_person_name_removal():
    """Presidio catches person names in freeform text."""
    from pipeline.presidio_scanner import PresidioScanner
    scanner = PresidioScanner()
    text = "Mike Albanese reported an issue with the account."
    masked, detections = scanner.scan_and_mask(text)
    assert "[PERSON]" in masked
    assert "Mike Albanese" not in masked
    assert any(d["entity_type"] == "PERSON" for d in detections)
    print("PASS: test_presidio_person_name_removal")

def test_presidio_organization_removal():
    """Presidio catches organization names."""
    from pipeline.presidio_scanner import PresidioScanner
    scanner = PresidioScanner()
    text = "Mosaic Advisors requested an update."
    masked, detections = scanner.scan_and_mask(text)
    assert "[ORG]" in masked or "Mosaic Advisors" not in masked
    print("PASS: test_presidio_organization_removal")

def test_text_normalization():
    """Cleaned text is lowercase and normalized."""
    result = preprocessor.process("Update the Addepar feed for Mike Albanese", task_id="norm1")
    # After Presidio + normalization
    assert result["cleaned_text"].islower()
    assert "Addepar" not in result["cleaned_text"]  # Should be removed by entity extraction
    assert "Mike Albanese" not in result["cleaned_text"]  # Should be masked by Presidio
    print("PASS: test_text_normalization")

def test_entity_extraction_on_normalized_text():
    """Entity extractor still works on lowercase text."""
    result = preprocessor.process("update feed in addepar for q1", task_id="norm2")
    # Input is lowercase, extractor is case-insensitive, output entities found
    assert any(e["entity"].lower() == "addepar" for e in result["entities"])
    assert result["primary_platform"] == "Addepar"  # Canonical form preserved
    print("PASS: test_entity_extraction_on_normalized_text")
```

---

### Task 7: Pipeline Execution

```bash
# Clear old outputs
Remove-Item outputs\* -Force

# Run pipeline
python preprocess.py

# Expected outputs:
# - outputs/corpus_clean.csv (with normalized cleaned_text)
# - outputs/anonymization_log.csv (entity extractions)
# - outputs/presidio_detections.csv (Presidio PII detections) [NEW]

# Verify
python analyze_corpus.py
```

**Success criteria:**
- Zero person names in cleaned_text
- Zero organization names (except platforms handled by entity extractor)
- All cleaned_text entries are lowercase
- Presidio audit log shows 200+ person name detections

---

## File Changes Summary

### New Files
1. `pipeline/presidio_scanner.py` - Presidio integration module
2. `outputs/presidio_detections.csv` - Audit log of Presidio detections

### Modified Files
1. `preprocess.py` - Add Presidio step + text normalization
2. `pipeline/preprocessor.py` - Add `normalize_text()` to final cleaning step
3. `tests/test_pipeline.py` - Add 4 new tests for Presidio + normalization
4. `README.md` or `SETUP.md` - Document Presidio installation

---

## Risk Assessment

### Low Risk
- ✅ Presidio is battle-tested Microsoft library
- ✅ Text normalization (lowercase) improves embedding quality
- ✅ Entity extractor already case-insensitive (no breaking changes)

### Medium Risk
- ⚠️ Presidio may be **too aggressive** (e.g., "General" detected as PERSON if sentence starts with it)
- ⚠️ Large spacy model (780MB) may be slow on 2,700 rows

### Mitigation
- Use confidence threshold (score >= 0.85) to reduce false positives
- Run Presidio in batch mode (100 rows at a time) with progress bar
- Make LOCATION masking optional (default: keep locations)

---

## Performance Considerations

**Presidio Speed:**
- ~0.1-0.5 seconds per text (depends on length)
- 2,704 rows × 0.2s = ~9 minutes total

**Optimization:**
- Batch processing (100 texts per batch)
- Progress bar for user feedback
- Cache spacy model (load once, reuse)

---

## Alternative Approach (NOT RECOMMENDED)

**Alternative:** Only use Presidio, remove regex-based PII

**Why not:**
- Regex patterns (email, phone, SSN) are 100% accurate and instant
- Presidio may miss edge cases (e.g., `[email@domain.com]` vs `email@domain.com`)
- Layered defense is safer: regex catches patterns, Presidio catches context-based PII

---

## Next Steps After This Plan

Once this plan is complete:
1. ✅ Corpus will have zero PII leakage (person names, orgs)
2. ✅ Corpus will be normalized (lowercase, no case-variant leakage)
3. ✅ Ready for Phase 2: Multi-resolution BERTopic discovery

**Phase 2 Preview:**
```bash
pip install sentence-transformers bertopic umap-learn
python discovery_scan.py  # Uses outputs/corpus_clean.csv
```

---

## Questions for User

1. **LOCATION masking:** Should we mask locations (Houston, TX, Oklahoma) or keep them as operational context?
   - Recommendation: Keep (not PII, useful context)

2. **Confidence threshold:** Presidio confidence >= 0.85 (default) or lower (0.7)?
   - Recommendation: 0.85 (reduces false positives)

3. **Organization names:** Mask ALL orgs (Mosaic Advisors, TOS) or only non-advisor orgs?
   - Current: Advisor orgs (TOS, Mosaic, Elevate) are in entities.yaml and removed by entity extractor
   - Recommendation: Let Presidio mask unknown orgs, entity extractor handles known ones
