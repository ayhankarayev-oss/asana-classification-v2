"""
add_subtype_definitions.py
===========================
Adds a 'Sub-type Definitions' section to leadership_report.html
with 3-sentence descriptions of each of the 20 sub-types.
Inserted just before the 'AFTER_SCATTER_SECTION_START' marker.
Does NOT modify any existing content.
"""
from pathlib import Path

REPORT_PATH = Path("outputs/reports/leadership_report.html")

DEFINITIONS_HTML = """
<!-- SUBTYPE_DEFINITIONS_SECTION_START -->
<div class="section" style="border-top: 4px solid #7c3aed; margin-top:40px;">
    <h2>Sub-type Definitions (20 Classes)</h2>
    <p style="color:#64748b; margin-bottom:15px; font-size:13px;">
        Short descriptions of what each sub-type cluster captures. Use these to evaluate or rename classes.
        Grouped by Issue Type.
    </p>

    <!-- Issue Type 1: Account & Data Onboarding -->
    <div style="margin-bottom:24px;">
        <h3 style="color:#4363d8; margin:0 0 12px 0; font-size:15px; border-bottom:2px solid #4363d8; padding-bottom:6px;">Account &amp; Data Onboarding</h3>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #4363d8;">
            <strong>Account Setup &amp; Connectivity</strong> <span style="color:#64748b; font-size:12px;">(304 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Setting up new client accounts AND resolving feed/connectivity issues on existing accounts. Only ~40% are genuinely new accounts; ~60% are fixing broken data feeds, establishing custodian API connections, and troubleshooting account sync failures. Covers the full lifecycle from initial account creation through ongoing connectivity maintenance.</p>
        </div>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #4363d8;">
            <strong>Historical Data Backfill</strong> <span style="color:#64748b; font-size:12px;">(63 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Importing historical statements, transactions, and positions for accounts that were recently connected or migrated. Includes validating import templates, uploading prior-period statements, and confirming data completeness after bulk loads. Typically a one-time effort per account during onboarding or platform migration.</p>
        </div>
    </div>

    <!-- Issue Type 2: Client Reporting & Deliverables -->
    <div style="margin-bottom:24px;">
        <h3 style="color:#911eb4; margin:0 0 12px 0; font-size:15px; border-bottom:2px solid #911eb4; padding-bottom:6px;">Client Reporting &amp; Deliverables</h3>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #911eb4;">
            <strong>Client Report &amp; View Management</strong> <span style="color:#64748b; font-size:12px;">(243 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Building, modifying, and maintaining client-facing reports and portal views in Addepar and other platforms. Covers creating custom views, adjusting report layouts, fixing display issues, and managing portal access configurations. Includes both new report development and ongoing maintenance of existing report templates.</p>
        </div>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #911eb4;">
            <strong>Invoicing &amp; Billing</strong> <span style="color:#64748b; font-size:12px;">(122 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Reviewing, generating, and troubleshooting client billing reports and fee schedules. Covers billing reviews for new clients, fee calculation validation, benchmark assignment changes that affect billing, and billing discussions with advisors. Tasks are typically recurring at quarter-end or when fee structures change.</p>
        </div>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #911eb4;">
            <strong>Investment Performance Analysis</strong> <span style="color:#64748b; font-size:12px;">(71 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Creating and validating performance benchmarks, return calculations, and performance attribution reports. Includes building blended benchmarks, investigating large cash returns that skew performance, and producing quarterly performance deliverables. Focused on the accuracy of return numbers shown to clients.</p>
        </div>
    </div>

    <!-- Issue Type 3: Data Quality & Governance -->
    <div style="margin-bottom:24px;">
        <h3 style="color:#f032e6; margin:0 0 12px 0; font-size:15px; border-bottom:2px solid #f032e6; padding-bottom:6px;">Data Quality &amp; Governance</h3>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #f032e6;">
            <strong>Missing Data &amp; Attributes</strong> <span style="color:#64748b; font-size:12px;">(203 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Identifying and filling in missing data fields such as asset sub-class, security names, account attributes, and model allocations. Covers gaps detected via automated workflows or manual review that prevent reports from rendering correctly. Reactive tasks triggered when something is incomplete in the system.</p>
        </div>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #f032e6;">
            <strong>Recurring Data Maintenance &amp; Audits</strong> <span style="color:#64748b; font-size:12px;">(215 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Regularly scheduled data quality checks, weekly audits, and systematic maintenance routines. Includes weekly client audits, compiling support tickets, reviewing fund inquiries, and running periodic data integrity checks. Proactive and scheduled (unlike Missing Data which is reactive).</p>
        </div>
    </div>

    <!-- Issue Type 4: Operational Administration -->
    <div style="margin-bottom:24px;">
        <h3 style="color:#808000; margin:0 0 12px 0; font-size:15px; border-bottom:2px solid #808000; padding-bottom:6px;">Operational Administration</h3>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #808000;">
            <strong>Access &amp; Permission Requests</strong> <span style="color:#64748b; font-size:12px;">(143 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Granting, modifying, or revoking user access to platforms, APIs, and client data. Covers credential setup, advisor directory updates, file sharing permissions, and onboarding new team members with proper access levels. Administrative tasks that control who can see/do what in the systems.</p>
        </div>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #808000;">
            <strong>Contact &amp; User Updates</strong> <span style="color:#64748b; font-size:12px;">(135 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Updating client contact information, user profiles, entity names, and person-level metadata across systems. Covers name changes, adding new contacts to groups, updating relationship mappings, and maintaining the people/entity directory. Focused on who people are rather than what they can access.</p>
        </div>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #808000;">
            <strong>Meetings, Reminders &amp; Follow-ups</strong> <span style="color:#64748b; font-size:12px;">(142 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Running and delivering scheduled reports, preparing materials for client meetings, and following up on outstanding items. Includes weekly report runs, quarterly deliverable packaging, dashboard updates before meetings, and reminder tasks. Administrative coordination rather than data work.</p>
        </div>
    </div>

    <!-- Issue Type 5: Ownership, Structure & Obligations -->
    <div style="margin-bottom:24px;">
        <h3 style="color:#42d4f4; margin:0 0 12px 0; font-size:15px; border-bottom:2px solid #42d4f4; padding-bottom:6px;">Ownership, Structure &amp; Obligations</h3>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #42d4f4;">
            <strong>Ownership &amp; Trust Structure</strong> <span style="color:#64748b; font-size:12px;">(139 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Modeling legal entity structures, ownership percentages, trust hierarchies, and dissolution of entities. Covers setting up ownership chains, updating beneficial ownership when structures change, and processing dissolved LLCs/trusts. Reflects the legal reality of who owns what through which vehicle.</p>
        </div>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #42d4f4;">
            <strong>Commitment &amp; Capital Call Tracking</strong> <span style="color:#64748b; font-size:12px;">(146 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Recording unfunded commitments, processing capital call notices, and tracking paid-in capital against total commitments. Covers uploading commitment schedules, investigating cash flow discrepancies in fund reports, and flagging when paid-in exceeds expected thresholds. Tied to PE/VC fund lifecycle obligations.</p>
        </div>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #42d4f4;">
            <strong>Debt &amp; Lending Instruments</strong> <span style="color:#64748b; font-size:12px;">(71 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Managing loan records, lines of credit, mortgage tracking, and direct lending fund positions. Covers updating loan balances, removing closed liabilities, processing lending fund distribution notices, and maintaining family loan schedules. The liability side of the balance sheet.</p>
        </div>
    </div>

    <!-- Issue Type 6: Portfolio & Investment Operations -->
    <div style="margin-bottom:24px;">
        <h3 style="color:#e6194b; margin:0 0 12px 0; font-size:15px; border-bottom:2px solid #e6194b; padding-bottom:6px;">Portfolio &amp; Investment Operations</h3>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #e6194b;">
            <strong>New Private Investment Setup</strong> <span style="color:#64748b; font-size:12px;">(247 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Creating and maintaining private investment records (PE, VC, real estate, hedge funds) including initial booking and ongoing attribute updates. Covers adding new private investments, updating existing fund details from client communications, and processing series/vintage changes. The full lifecycle from first booking to exit.</p>
        </div>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #e6194b;">
            <strong>Security &amp; Position Updates</strong> <span style="color:#64748b; font-size:12px;">(102 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Modifying security-level attributes like entity modeling, ticker renames, price overrides, and corporate action processing. Covers restructuring how a security is represented in the system, updating ownership percentages at tranche level, and fixing position discrepancies. Focused on the security master and how positions are modeled.</p>
        </div>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #e6194b;">
            <strong>Position Cleanup &amp; Deduplication</strong> <span style="color:#64748b; font-size:12px;">(70 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Removing duplicate positions, hiding stale online holdings, merging split positions, and cleaning up data artifacts. Covers deleting phantom positions, merging transaction history when positions are consolidated, and modifying open dates on merged positions. Housekeeping to keep the portfolio view accurate.</p>
        </div>
    </div>

    <!-- Issue Type 7: Transaction & Cash Flow Processing -->
    <div style="margin-bottom:24px;">
        <h3 style="color:#f58231; margin:0 0 12px 0; font-size:15px; border-bottom:2px solid #f58231; padding-bottom:6px;">Transaction &amp; Cash Flow Processing</h3>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #f58231;">
            <strong>Cash Flow &amp; Distribution Management</strong> <span style="color:#64748b; font-size:12px;">(75 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Recording and verifying cash inflows including distributions received, income payments, mortgage transactions, and capital call funding. Covers confirming distribution amounts against statements, processing monthly mortgage payments, and booking cash events correctly. The "money came in, verify it, book it" workflow — broader than just distributions.</p>
        </div>
    </div>

    <!-- Issue Type 8: Valuation & Pricing -->
    <div style="margin-bottom:24px;">
        <h3 style="color:#3cb44b; margin:0 0 12px 0; font-size:15px; border-bottom:2px solid #3cb44b; padding-bottom:6px;">Valuation &amp; Pricing</h3>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #3cb44b;">
            <strong>Scheduled Valuation Feeds</strong> <span style="color:#64748b; font-size:12px;">(81 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Automated or semi-automated valuation imports that run on a regular schedule (weekly/monthly). Majority are batch "valuation import" tasks with dates, plus ad-hoc valuation update requests for specific REITs or funds. Routine pipeline work that keeps asset prices current.</p>
        </div>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #3cb44b;">
            <strong>Periodic NAV &amp; Valuation Entry</strong> <span style="color:#64748b; font-size:12px;">(56 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Routine manual data entry to update dollar values or percentage returns from client-provided statements. Includes updating gold prices, life insurance values, farm share valuations, and processing brokerage statements shared via email. Monthly/quarterly recurring tasks — the human enters the number that a feed can't provide automatically.</p>
        </div>

        <div style="margin-bottom:12px; padding:12px 16px; background:#f8fafc; border-radius:6px; border-left:3px solid #3cb44b;">
            <strong>Cost Basis Reconciliation</strong> <span style="color:#64748b; font-size:12px;">(55 docs)</span>
            <p style="margin:6px 0 0 0; font-size:13px;">Resolving missing or incorrect cost basis data, performing instance-wide cost basis cleanups, and reconciling unfunded/vintage adjustments. Covers comparing cost basis against custodian records, reclassifying items, and importing fund quarterly adjustment data. Critical for accurate gain/loss reporting.</p>
        </div>
    </div>
</div>
<!-- SUBTYPE_DEFINITIONS_SECTION_END -->
"""


def main():
    content = REPORT_PATH.read_text(encoding="utf-8")

    # Check if already present
    if "SUBTYPE_DEFINITIONS_SECTION_START" in content:
        # Replace existing
        start = content.find("<!-- SUBTYPE_DEFINITIONS_SECTION_START -->")
        end = content.find("<!-- SUBTYPE_DEFINITIONS_SECTION_END -->") + len("<!-- SUBTYPE_DEFINITIONS_SECTION_END -->")
        content = content[:start] + DEFINITIONS_HTML + content[end:]
        print("Replaced existing Sub-type Definitions section.")
    else:
        # Insert before AFTER_SCATTER_SECTION_START
        marker = "<!-- AFTER_SCATTER_SECTION_START -->"
        idx = content.find(marker)
        if idx < 0:
            print("ERROR: Could not find AFTER_SCATTER_SECTION_START marker")
            return
        content = content[:idx] + DEFINITIONS_HTML + "\n" + content[idx:]
        print("Inserted Sub-type Definitions section before Section 8.")

    REPORT_PATH.write_text(content, encoding="utf-8")
    size_kb = REPORT_PATH.stat().st_size / 1024
    print(f"File size: {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
