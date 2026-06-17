"""Trim the Taxonomy Refinement section to only T0 with concise description."""
from pathlib import Path

REPORT_PATH = Path(r"c:\Users\Lenovo\Desktop\asana-classification-v2\outputs\reports\leadership_report.html")

SECTION_HTML = """
<!-- TAXONOMY_REFINEMENT_START -->
<div class="section" style="border-top: 4px solid #2563eb;">
    <h2>Cluster Performance & Taxonomy Refinement</h2>

    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:20px; margin:15px 0;">
        <h3 style="color:#1e293b; margin-bottom:10px;">T0: "New Account Onboarding" → "Account Setup & Connectivity"</h3>
        <p style="font-size:13px; color:#475569;">
            <strong>290 docs | 10.8%</strong> — Only ~40% of tasks are genuinely new accounts. 
            The majority (~60%) involve existing accounts with feed/connectivity issues 
            ("accounts not showing up", "not receiving data", "ensure accounts are online/feeding") 
            or account maintenance (balance discrepancies, master reassignments). 
            Renamed to reflect the full scope. Not split (silhouette &lt; 0.3 — same team handles both).
        </p>
    </div>
</div>
<!-- TAXONOMY_REFINEMENT_END -->
"""

with open(REPORT_PATH, "r", encoding="utf-8") as f:
    html = f.read()

# Remove previous version
start = "<!-- TAXONOMY_REFINEMENT_START -->"
end = "<!-- TAXONOMY_REFINEMENT_END -->"
if start in html:
    html = html[:html.index(start)] + html[html.index(end) + len(end):]

# Insert before split validation or footer
split = "<!-- SPLIT_VALIDATION_START -->"
pos = html.index(split) if split in html else html.rfind('<div class="footer">')
html = html[:pos] + SECTION_HTML + "\n" + html[pos:]

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("Done. Concise T0 refinement section injected.")
