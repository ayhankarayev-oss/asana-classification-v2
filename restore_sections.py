"""
restore_sections.py
====================
Restores the 'Cluster Performance & Taxonomy Refinement' and 
'SPLIT VALIDATION REPORT' sections that were accidentally removed.
Inserts them between the ANALYSIS_SECTION_END marker and the footer.
"""
from pathlib import Path

REPORT_PATH = Path("outputs/reports/leadership_report.html")

CLUSTER_PERFORMANCE_HTML = """
<!-- CLUSTER_PERFORMANCE_SECTION_START -->
<div class="section" style="border-top: 4px solid #f59e0b; margin-top:40px;">
    <h2>Cluster Performance &amp; Taxonomy Refinement</h2>
    <p style="color:#64748b; margin-bottom:15px; font-size:13px;">Detailed analysis of each cluster rename, merge, or status decision. Based on 30-50 sample reviews per cluster.</p>

    <div style="margin-bottom:20px; padding:16px; background:#fffbeb; border-left:4px solid #f59e0b; border-radius:4px;">
        <h4 style="margin:0 0 8px 0; color:#92400e;">T0: &rarr; &ldquo;Account Setup &amp; Connectivity&rdquo;</h4>
        <p style="margin:0; font-size:13px;">290 docs &mdash; Only ~40% are new accounts. ~60% are feed/connectivity issues on existing accounts. Renamed to reflect full scope. Not split (silhouette &lt; 0.3).</p>
    </div>

    <div style="margin-bottom:20px; padding:16px; background:#f0fdf4; border-left:4px solid #16a34a; border-radius:4px;">
        <h4 style="margin:0 0 8px 0; color:#166534;">T7 + T17: Merged &rarr; &ldquo;Client Report &amp; View Management&rdquo;</h4>
        <p style="margin:0; font-size:13px;">240 docs &mdash; Both clusters contained report/view tasks (maintenance + development). Merged under platform-agnostic name.</p>
    </div>

    <div style="margin-bottom:20px; padding:16px; background:#fffbeb; border-left:4px solid #f59e0b; border-radius:4px;">
        <h4 style="margin:0 0 8px 0; color:#92400e;">T1: &rarr; &ldquo;Private Investment Lifecycle&rdquo;</h4>
        <p style="margin:0; font-size:13px;">244 docs &mdash; Only ~30% are NEW bookings. Majority are updates, research, exits, attribute fixes on existing PE/VC/RE investments. &ldquo;Lifecycle&rdquo; captures the full range.</p>
    </div>

    <div style="margin-bottom:20px; padding:16px; background:#f8fafc; border-left:4px solid #64748b; border-radius:4px;">
        <h4 style="margin:0 0 8px 0; color:#334155;">T8: &ldquo;Security &amp; Position Updates&rdquo; (unchanged)</h4>
        <p style="margin:0; font-size:13px;">96 docs &mdash; Entity modeling, series closings, down rounds, ticker renames, price overrides. Name kept as-is per review.</p>
    </div>

    <div style="margin-bottom:20px; padding:16px; background:#fffbeb; border-left:4px solid #f59e0b; border-radius:4px;">
        <h4 style="margin:0 0 8px 0; color:#92400e;">T20: &rarr; &ldquo;Periodic NAV &amp; Valuation Entry&rdquo;</h4>
        <p style="margin:0; font-size:13px;">52 docs &mdash; Routine data entry: update a $ value or % return from client-provided statements. Monthly/quarterly recurring.</p>
    </div>

    <div style="margin-bottom:20px; padding:16px; background:#fffbeb; border-left:4px solid #f59e0b; border-radius:4px;">
        <h4 style="margin:0 0 8px 0; color:#92400e;">T16: &rarr; &ldquo;Cash Flow &amp; Distribution Management&rdquo;</h4>
        <p style="margin:0; font-size:13px;">72 docs &mdash; Recording and verifying cash inflows (distributions, income, mortgage payments, cap calls). Action: money came in &rarr; book it correctly.</p>
    </div>

    <div style="margin-bottom:20px; padding:16px; background:#eff6ff; border-left:4px solid #2563eb; border-radius:4px;">
        <h4 style="margin:0 0 8px 0; color:#1e40af;">T1 + T8 + T20: Relationship Note (Investment Lifecycle)</h4>
        <p style="margin:0; font-size:13px;">These three cover different lifecycle stages: T1=CREATE, T8=MODIFY, T20=VALUE. Flagged as potential merge candidates. Will be validated using hierarchical classification tree before final decision.</p>
    </div>
</div>
<!-- CLUSTER_PERFORMANCE_SECTION_END -->
"""

SPLIT_VALIDATION_HTML = """
<!-- SPLIT_VALIDATION_SECTION_START -->
<div class="section" style="border-top: 4px solid #dc2626; margin-top:30px;">
    <h2>SPLIT VALIDATION REPORT</h2>
    <p style="color:#64748b; margin-bottom:15px; font-size:13px;">Only clusters with silhouette score &gt; 0.3 are shown below (bimodal clusters). Review each split proposal and decide whether it improves classification accuracy.<br><span style="color:#dc2626;">Red = Sub-cluster A</span>, <span style="color:#16a34a;">Green = Sub-cluster B</span>.</p>

    <p style="font-weight:bold; margin:15px 0;">2 split candidate(s) found:</p>

    <!-- Split Candidate 1 -->
    <div style="margin-bottom:30px; padding:20px; background:#fef2f2; border:1px solid #fca5a5; border-radius:8px;">
        <h3 style="margin:0 0 10px 0; color:#dc2626;">Scheduled Valuation Feeds (81 docs)</h3>
        <p style="margin:0 0 12px 0; font-size:14px;"><strong>Silhouette Score: 0.656</strong> (STRONG split signal)</p>

        <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
            <div style="background:white; padding:12px; border-radius:6px; border-left:4px solid #dc2626;">
                <h4 style="margin:0 0 6px 0; color:#dc2626;">Sub-cluster A (22 docs)</h4>
                <p style="font-size:12px; color:#666; margin:0 0 8px 0;"><strong>Keywords:</strong> valuation, update, valuation update, valuations, value, 31</p>
                <p style="font-size:12px; margin:0;"><strong>Samples:</strong></p>
                <ul style="font-size:11px; margin:4px 0 0 16px; padding:0;">
                    <li>[1212419572152824] vinebrook reit valuation updates...</li>
                    <li>[1213683806817212] exit date and valuation date...</li>
                    <li>[1213633621968349] latest valuation date...</li>
                </ul>
            </div>
            <div style="background:white; padding:12px; border-radius:6px; border-left:4px solid #16a34a;">
                <h4 style="margin:0 0 6px 0; color:#16a34a;">Sub-cluster B (59 docs)</h4>
                <p style="font-size:12px; color:#666; margin:0 0 8px 0;"><strong>Keywords:</strong> valuation, import, 2026, 05, valuation import, import 2026</p>
                <p style="font-size:12px; margin:0;"><strong>Samples:</strong></p>
                <ul style="font-size:11px; margin:4px 0 0 16px; padding:0;">
                    <li>[1215037646722409] valuation import | 2026-05-21...</li>
                    <li>[1214817389401299] valuation import | 2026-05-14...</li>
                    <li>[1214621550868255] valuation import | 2026-05-07...</li>
                </ul>
            </div>
        </div>
        <p style="margin:12px 0 0 0; font-size:12px; color:#92400e; font-style:italic;"><strong>YOUR DECISION:</strong> Should this cluster be split into two separate classes? If yes, suggest names for Sub-A and Sub-B.</p>
    </div>

    <!-- Split Candidate 2 -->
    <div style="margin-bottom:30px; padding:20px; background:#fffbeb; border:1px solid #fcd34d; border-radius:8px;">
        <h3 style="margin:0 0 10px 0; color:#b45309;">Cost Basis Reconciliation (55 docs)</h3>
        <p style="margin:0 0 12px 0; font-size:14px;"><strong>Silhouette Score: 0.364</strong> (Moderate split signal)</p>

        <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
            <div style="background:white; padding:12px; border-radius:6px; border-left:4px solid #dc2626;">
                <h4 style="margin:0 0 6px 0; color:#dc2626;">Sub-cluster A (33 docs)</h4>
                <p style="font-size:12px; color:#666; margin:0 0 8px 0;"><strong>Keywords:</strong> cost, basis, cost basis, client, missing, missing cost</p>
                <p style="font-size:12px; margin:0;"><strong>Samples:</strong></p>
                <ul style="font-size:11px; margin:4px 0 0 16px; padding:0;">
                    <li>[1213602032277600] tpl cost basis x2005...</li>
                    <li>[1212872920656841] [client] instance wide cost basis cleanup...</li>
                    <li>[1205538026541898] please review missing cost basis for the entire household and let us know what is needed to resolve...</li>
                </ul>
            </div>
            <div style="background:white; padding:12px; border-radius:6px; border-left:4px solid #16a34a;">
                <h4 style="margin:0 0 6px 0; color:#16a34a;">Sub-cluster B (22 docs)</h4>
                <p style="font-size:12px; color:#666; margin:0 0 8px 0;"><strong>Keywords:</strong> vintage, unfunded, client, fund, expense, vi</p>
                <p style="font-size:12px; margin:0;"><strong>Samples:</strong></p>
                <ul style="font-size:11px; margin:4px 0 0 16px; padding:0;">
                    <li>[1214246584462375] please reclassify the items outlined below in the task description...</li>
                    <li>[1213422463898816] unfunded adjustment import for [client]...</li>
                    <li>[1209309480770508] fund quarterly update - stolar [client] &amp; stolar fund adjustment...</li>
                </ul>
            </div>
        </div>
        <p style="margin:12px 0 0 0; font-size:12px; color:#92400e; font-style:italic;"><strong>YOUR DECISION:</strong> Should this cluster be split into two separate classes? If yes, suggest names for Sub-A and Sub-B.</p>
    </div>
</div>
<!-- SPLIT_VALIDATION_SECTION_END -->
"""


def main():
    content = REPORT_PATH.read_text(encoding="utf-8")

    # Check if already present
    if "CLUSTER_PERFORMANCE_SECTION_START" in content:
        print("Cluster Performance section already exists. Skipping.")
        return
    if "SPLIT_VALIDATION_SECTION_START" in content:
        print("Split Validation section already exists. Skipping.")
        return

    # Insert after ANALYSIS_SECTION_END
    end_marker = "<!-- ANALYSIS_SECTION_END -->"
    end_idx = content.find(end_marker)
    if end_idx < 0:
        print("ERROR: Could not find ANALYSIS_SECTION_END marker")
        return

    end_idx += len(end_marker)

    new_content = (
        content[:end_idx]
        + CLUSTER_PERFORMANCE_HTML
        + SPLIT_VALIDATION_HTML
        + content[end_idx:]
    )

    REPORT_PATH.write_text(new_content, encoding="utf-8")
    size_kb = REPORT_PATH.stat().st_size / 1024
    print(f"Done! Restored both sections.")
    print(f"File size: {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
