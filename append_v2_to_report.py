"""
append_v2_to_report.py
========================
Extracts the v2 dashboard content and appends it as a new section
at the bottom of leadership_report.html (before the closing tags).
Does NOT modify any existing content in the original report.
"""
from pathlib import Path

V2_PATH = Path("outputs/reports/leadership_report_v2.html")
REPORT_PATH = Path("outputs/reports/leadership_report.html")


def main():
    print("=" * 60)
    print("APPEND V2 CONTENT TO ORIGINAL REPORT")
    print("=" * 60)

    v2_content = V2_PATH.read_text(encoding="utf-8")
    report_content = REPORT_PATH.read_text(encoding="utf-8")

    # Check if already appended
    if "V2_REGROUPED_SECTION_START" in report_content:
        # Replace existing
        start = report_content.find("<!-- V2_REGROUPED_SECTION_START -->")
        end = report_content.find("<!-- V2_REGROUPED_SECTION_END -->") + len("<!-- V2_REGROUPED_SECTION_END -->")
        report_content = report_content[:start] + "{PLACEHOLDER}" + report_content[end:]
        print("  Replacing existing v2 section...")
    else:
        # Find insertion point: before </div></body></html> at the very end
        # Insert after the AFTER_SCATTER_SECTION_END
        marker = "<!-- AFTER_SCATTER_SECTION_END -->"
        idx = report_content.find(marker)
        if idx > 0:
            idx += len(marker)
            report_content = report_content[:idx] + "\n{PLACEHOLDER}\n" + report_content[idx:]
        else:
            # Fallback: before closing body
            idx = report_content.rfind("</body>")
            report_content = report_content[:idx] + "\n{PLACEHOLDER}\n" + report_content[idx:]
        print("  Inserting v2 section after Section 8...")

    # Extract v2 body (between <header> and footer closing)
    v2_start = v2_content.find('<header>')
    v2_end = v2_content.find('<div class="footer">')
    v2_body = v2_content[v2_start:v2_end]

    # Wrap in a clearly marked section
    v2_section = f"""
<!-- V2_REGROUPED_SECTION_START -->
<div style="border-top: 6px solid #0f3460; margin-top:60px; padding-top:40px;">
    <div style="background: linear-gradient(135deg, #0f3460 0%, #16213e 100%); color:white; padding:30px 40px; text-align:center; border-radius:12px; margin-bottom:30px; box-shadow:0 4px 20px rgba(0,0,0,0.15);">
        <h1 style="font-size:26px; font-weight:700; margin-bottom:6px;">Phase 9: Regrouped Taxonomy (v2)</h1>
        <p style="color:#a0aec0; font-size:13px;">8 Issue Types × 19 Sub-types | Refined from v1 based on semantic review</p>
    </div>

{v2_body}

</div>
<!-- V2_REGROUPED_SECTION_END -->
"""

    # Replace placeholder
    report_content = report_content.replace("{PLACEHOLDER}", v2_section)

    REPORT_PATH.write_text(report_content, encoding="utf-8")
    size_kb = REPORT_PATH.stat().st_size / 1024
    print(f"\n  Saved: {REPORT_PATH}")
    print(f"  Size: {size_kb:.0f} KB")
    print("  Done! v2 content appended as new section at bottom.")
    print("=" * 60)


if __name__ == "__main__":
    main()
