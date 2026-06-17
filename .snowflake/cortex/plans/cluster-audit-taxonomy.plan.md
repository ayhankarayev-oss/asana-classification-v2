---
name: "cluster-audit-taxonomy"
created: "2026-06-15T15:39:22.132Z"
status: pending
---

### Phase 1: The "Audit Passport" Generation (The Script)

# - **Goal:** Build a script that extracts the "DNA" of each cluster.

- **Action:** Build `audit_taxonomy.py` using dynamic file discovery.

- **The Output:** For each of the 23 clusters, the script will generate a "Passport" containing:

  - **Statistical DNA:** Keywords, size, and coherence score.

  - **Operational Context:** Platform distribution (Addepar vs. Arch vs. Egnyte).

  - **Edge Case List:** The bottom 5% of documents (distance-based) to be flagged for review.

  - **Merge/Split Recommendations:** Based on similarity matrix thresholds (> 0.75) and silhouette scores (for splits).

### Phase 2: The "AI-Driven" Naming & Validation

# - **Goal:** Convert statistical output into business-ready taxonomy labels.

- **Action:** Instead of hard-coding rules, feed the "Cluster Passports" into your LLM (Cortex).

- **The Prompt:** "Here is the statistical profile for Cluster {X}. Does the current auto-label match the keywords and samples? Please suggest a professional business name and identify if any listed 'Edge Case' tasks should be moved to a different cluster."

- **Benefit:** This uses the AI as an analyst, preventing the "brittle code" trap and ensuring naming consistency.

### Phase 3: Consolidation & Taxonomy Synthesis

# - **Goal:** Create the final "5-Molecule" business taxonomy.

- **Action:**

  - suggest which could be merged or spilt based on the AI analyse of each task in the file make sure that it covers all the 2700 tasks make sure that whether it is worth beign classified like that or not? 

  - **Pruning:** Remove the edge cases identified as "Low Confidence" or "Noise."

  - **Final Taxonomy:** Assemble the `taxonomy_draft.csv` (Topic ID → Business Pillar → Specific Name).

### Phase 4: Leadership Presentation (The "Quality" Scorecard)

# - **Goal:** Prove to your boss that this model is stable and accurate.

- **Action:**

  - **Visual 1 (The Separation):** Use `topic_model.visualize_topics()` to show the Inter-Topic Distance Map (bubbles).

  - **Visual 2 (The Operational Heatmap):** Show the `Topic vs. Platform` distribution heatmap to prove you know *where* the work is originating.

  - **Visual 3 (The Audit Scorecard):** Present the "5-Molecule Taxonomy" with its associated Confidence scores.

### Verification Checklist (Safety Checks)

# 1. **Traceability:** Every row in `taxonomy_draft.csv` must be traceable to an `asana_gid` in `corpus_clean.csv`.

2. **No Automated Destruction:** Do not automate "Splits" or "Merges." Let the script *recommend* them, and you *execute* them. This protects you from the AI making a mistake that you cannot easily reverse.

3. **Stability:** Because you are using cached embeddings, the script will be very fast, allowing you to re-run the audit as many times as you like while you iterate on names.

