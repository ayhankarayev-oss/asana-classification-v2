from pathlib import Path

REPORT_PATH = Path(r"c:\Users\Lenovo\Desktop\asana-classification-v2\outputs\reports\leadership_report.html")

SECTION = """
<!-- TAXONOMY_REFINEMENT_START -->
<div class="section" style="border-top: 4px solid #2563eb;">
    <h2>Cluster Performance &amp; Taxonomy Refinement</h2>

    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:20px; margin:15px 0;">
        <h3 style="color:#1e293b; margin-bottom:10px;">T0: Renamed to "Account Setup &amp; Connectivity"</h3>
        <p style="font-size:13px; color:#475569;">
            <strong>290 docs | 10.8%</strong> &mdash; Only ~40% are genuinely new accounts.
            The majority (~60%) involve existing accounts with feed/connectivity issues
            or maintenance (balance discrepancies, master reassignments).
            Renamed to reflect full scope. Not split (silhouette &lt; 0.3).
        </p>
    </div>

    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:20px; margin:15px 0;">
        <h3 style="color:#1e293b; margin-bottom:10px;">T1 + T8 + T20: Potential Merge (Investment Lifecycle)</h3>
        <p style="font-size:13px; color:#475569;">
            These three clusters cover different stages of the same lifecycle:
            <strong>T1</strong> = CREATE (book new investment, 244 docs),
            <strong>T8</strong> = MODIFY (structural changes, 96 docs),
            <strong>T20</strong> = VALUE (update dollar amounts, 52 docs).
        </p>
        <p style="font-size:13px; color:#475569; margin-top:8px;">
            <strong>Status:</strong> Flagged as potential merge candidates.
            Will be validated using a hierarchical classification tree to determine
            if they should remain separate sub-types under a shared parent pillar
            or be collapsed into a single class.
        </p>
    </div>
</div>
<!-- TAXONOMY_REFINEMENT_END -->
"""

with open(REPORT_PATH, "r", encoding="utf-8") as f:
    html = f.read()

start = "<!-- TAXONOMY_REFINEMENT_START -->"
end = "<!-- TAXONOMY_REFINEMENT_END -->"
if start in html:
    html = html[:html.index(start)] + html[html.index(end) + len(end):]

split_marker = "<!-- SPLIT_VALIDATION_START -->"
if split_marker in html:
    pos = html.index(split_marker)
else:
    pos = html.rfind('<div class="footer">')

html = html[:pos] + SECTION + "\n" + html[pos:]

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("Done. Concise refinement section injected (T0 rename + T1/T8/T20 merge note).")
