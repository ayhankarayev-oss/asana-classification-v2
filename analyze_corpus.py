"""
Comprehensive line-by-line corpus analysis for PII and sensitive data leakage.
Checks each cleaned_text entry for: emails, phones, person names, client names, 
platform names, and other sensitive patterns.
"""
import pandas as pd
import re
from collections import defaultdict

# Known entities that should NOT appear in cleaned_text
PLATFORMS = [
    'addepar', 'arch', 'egnyte', 'orca', 'venn', 'tetrix', 'bridge', 'overstone',
    'nines', 'sharepoint', 'plaid', 'byall', 'epicc', 'salesforce', 'sfdc',
    'sage', 'intacct', 'knowledger', 'quickbooks', 'asana'
]

CUSTODIANS = [
    'goldman sachs', 'goldman', 'morgan stanley', 'raymond james', 'charles schwab',
    'schwab', 'fidelity', 'pershing', 'northern trust', 'stifel',
    'interactive brokers', 'ibkr', 'merrill lynch', 'merrill', 'edward jones',
    'vanguard', 'blackrock', 'ubs', 'jefferies'
]

BANKS = [
    'bank of america', 'wells fargo', 'jpmorgan', 'bofa', 'citibank', 'citigroup',
    'citi', 'regions bank', 'td bank', 'us bank'
]

CLIENT_NAMES = [
    'trousdale', 'sarosphere', 'schmulen', 'dorsar', 'parkview', 'goradia',
    'woodland', 'annunziato', 'boelte', 'mussafer', 'outwing', 'bellco',
    'sweetland', 'buryakovsky', 'peregrine', 'willica', 'ialumbra',
    'golden', 'weinstock', 'webster', 'stern', 'connery', 'mosaic', 'elevate',
    'pjs', 'awb', 'nella', 'dume', 'origin', 'epicc'
]

# Regex patterns for PII
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_PATTERN = re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')

# Common person name patterns (capitalized words that might be names)
def detect_person_names(text):
    """Detect potential person names (capitalized words in certain contexts)."""
    # Look for patterns like "John Smith" or "Dr. Jane Doe"
    name_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
    matches = name_pattern.findall(text)
    # Filter out common false positives
    false_positives = {'General Updates', 'New Investment', 'Private Investment', 
                      'Trust Account', 'Family Office', 'Asset Class', 'North America',
                      'South State', 'West Coast', 'East Coast', 'New York', 'Los Angeles'}
    return [m for m in matches if m not in false_positives and len(m.split()) >= 2]

def check_text(text, row_id):
    """Check a single text entry for all types of leakage."""
    issues = []
    text_lower = text.lower()
    
    # 1. Email addresses
    emails = EMAIL_PATTERN.findall(text)
    if emails:
        for email in emails:
            if '[EMAIL]' not in email.upper():  # Skip if already masked
                issues.append(('EMAIL', email))
    
    # 2. Phone numbers
    phones = PHONE_PATTERN.findall(text)
    if phones:
        for phone in phones:
            issues.append(('PHONE', phone))
    
    # 3. SSN
    ssns = SSN_PATTERN.findall(text)
    if ssns:
        for ssn in ssns:
            issues.append(('SSN', ssn))
    
    # 4. Credit cards
    cards = CREDIT_CARD_PATTERN.findall(text)
    if cards:
        for card in cards:
            issues.append(('CREDIT_CARD', card))
    
    # 5. Platform names
    for platform in PLATFORMS:
        if platform == 'arch':
            # Special handling for Arch (avoid 'architecture', 'archive')
            pattern = r'\barch(?!\w)'
        else:
            pattern = r'\b' + re.escape(platform) + r'\b'
        if re.search(pattern, text_lower):
            issues.append(('PLATFORM', platform))
    
    # 6. Custodian names
    for cust in CUSTODIANS:
        if re.search(r'\b' + re.escape(cust) + r'\b', text_lower):
            issues.append(('CUSTODIAN', cust))
    
    # 7. Bank names
    for bank in BANKS:
        if re.search(r'\b' + re.escape(bank) + r'\b', text_lower):
            issues.append(('BANK', bank))
    
    # 8. Client names
    for client in CLIENT_NAMES:
        if re.search(r'\b' + re.escape(client) + r'\b', text_lower):
            issues.append(('CLIENT', client))
    
    # 9. Person names (heuristic)
    person_names = detect_person_names(text)
    for name in person_names:
        # Check if it's already masked
        if '[EMPLOYEE]' not in text and '[CLIENT]' not in text:
            issues.append(('PERSON_NAME', name))
    
    # 10. Boilerplate
    boilerplate_phrases = [
        'how may we assist you', 'submitted through', 'priority level',
        'please select', 'please specify', 'additional information'
    ]
    for phrase in boilerplate_phrases:
        if phrase in text_lower:
            issues.append(('BOILERPLATE', phrase))
    
    return issues


def main():
    """Run comprehensive analysis on corpus."""
    print("=" * 80)
    print("COMPREHENSIVE CORPUS ANALYSIS - Line by Line PII/Leakage Detection")
    print("=" * 80)
    
    # Load corpus
    df = pd.read_csv(r'c:\Users\Lenovo\Desktop\asana-classification-v2\outputs\corpus_clean.csv',
                     dtype=str, keep_default_na=False)
    
    print(f"\nAnalyzing {len(df)} rows...\n")
    
    # Track all issues by type
    issues_by_type = defaultdict(list)
    rows_with_issues = set()
    
    # Analyze each row
    for idx, row in df.iterrows():
        task_id = row['task_id']
        cleaned_text = row['cleaned_text']
        
        issues = check_text(cleaned_text, task_id)
        
        if issues:
            rows_with_issues.add(idx)
            for issue_type, value in issues:
                issues_by_type[issue_type].append({
                    'row': idx,
                    'task_id': task_id,
                    'value': value,
                    'context': cleaned_text[:100]  # First 100 chars for context
                })
    
    # Report findings
    print(f"SUMMARY: {len(rows_with_issues)} rows with potential issues out of {len(df)}")
    print(f"Clean rate: {(len(df) - len(rows_with_issues)) / len(df) * 100:.1f}%\n")
    
    if not issues_by_type:
        print("✓ NO ISSUES FOUND - Corpus is clean!")
        return
    
    print("\n" + "=" * 80)
    print("DETAILED FINDINGS BY CATEGORY")
    print("=" * 80)
    
    for issue_type in sorted(issues_by_type.keys()):
        issues = issues_by_type[issue_type]
        print(f"\n{issue_type}: {len(issues)} occurrences")
        print("-" * 80)
        
        # Show first 10 examples
        for i, issue in enumerate(issues[:10]):
            print(f"  [{i+1}] Row {issue['row']} | task_id={issue['task_id']}")
            print(f"      Found: '{issue['value']}'")
            print(f"      Context: {issue['context'][:80]}...")
        
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    
    # Export detailed report
    report_path = r'c:\Users\Lenovo\Desktop\asana-classification-v2\outputs\leakage_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("FULL LEAKAGE REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for issue_type in sorted(issues_by_type.keys()):
            issues = issues_by_type[issue_type]
            f.write(f"\n{issue_type}: {len(issues)} occurrences\n")
            f.write("-" * 80 + "\n")
            
            for issue in issues:
                f.write(f"Row {issue['row']} | task_id={issue['task_id']}\n")
                f.write(f"  Value: {issue['value']}\n")
                f.write(f"  Context: {issue['context']}\n\n")
    
    print(f"\n\n✓ Full report saved to: {report_path}")


if __name__ == '__main__':
    main()
