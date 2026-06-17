"""Post-hardening bias verification."""
import pandas as pd
import re

df = pd.read_csv(r'c:\Users\Lenovo\Desktop\asana-classification-v2\outputs\corpus_clean.csv', dtype=str, keep_default_na=False)

print("=== PLATFORM LEAKAGE CHECK ===")
platforms = ['addepar', 'arch', 'egnyte', 'orca', 'venn', 'schwab', 'fidelity',
             'goldman', 'morgan stanley', 'jpmorgan', 'wells fargo', 'plaid',
             'epicc', 'salesforce', 'quickbooks', 'knowledger', 'sharepoint']
leaked = 0
for p in platforms:
    pat = r'\b' + re.escape(p) + r'\b' if p != 'arch' else r'\barch(?![a-z])'
    count = df['cleaned_text'].str.lower().str.contains(pat, regex=True).sum()
    if count > 0:
        print(f"  LEAK: '{p}' in {count} docs")
        leaked += count
if leaked == 0:
    print("  CLEAN: Zero platform names in cleaned_text")

print("\n=== FORM BOILERPLATE CHECK ===")
boilerplate = ['How may we assist you', 'Please select the priority',
               'Please specify', 'priority level for this request',
               'additional information you would like']
bp_total = 0
for bp in boilerplate:
    count = df['cleaned_text'].str.contains(bp, case=False, regex=False).sum()
    if count > 0:
        print(f"  LEAK: '{bp}' in {count} docs")
        bp_total += count
if bp_total == 0:
    print("  CLEAN: Zero form boilerplate in cleaned_text")

print("\n=== FORM-TOKEN FREQUENCY (before vs after comparison) ===")
form_tokens = ['select', 'priority', 'level', 'submitted', 'assist']
for tok in form_tokens:
    count = df['cleaned_text'].str.contains(r'\b' + tok + r'\b', case=False, regex=True).sum()
    pct = count / len(df) * 100
    print(f"  '{tok}': {count} docs ({pct:.1f}%)")

print("\n=== HOUSEHOLD NAME LEAKAGE CHECK ===")
# Check known large households
test_names = ['Trousdale', 'Dorsar', 'Parkview', 'Dume', 'Schmulen', 'Annunziato',
              'Goradia', 'Woodland', 'Boelte', 'Outwing']
for name in test_names:
    count = df['cleaned_text'].str.contains(r'\b' + re.escape(name) + r'\b', case=False, regex=True).sum()
    if count > 0:
        print(f"  LEAK: '{name}' in {count} docs")
    else:
        print(f"  OK: '{name}' removed")

print("\n=== TOP 20 TOKENS (operational signal) ===")
stopwords = {'client','employee','email','phone','please','that','this','with','from','have',
             'been','will','they','their','into','task','name','also','would','should','could',
             'about','which','there','were','what','when','your','more','make','need','like',
             'just','some','than','other','each','does','very','these','those','then','only'}
from collections import Counter
all_tokens = []
for text in df['cleaned_text']:
    for t in re.findall(r'[A-Za-z]{4,}', text):
        if t.lower() not in stopwords:
            all_tokens.append(t.lower())
top20 = Counter(all_tokens).most_common(20)
for tok, cnt in top20:
    doc_count = df['cleaned_text'].str.contains(r'\b' + re.escape(tok) + r'\b', case=False, regex=True).sum()
    print(f"  {tok:<20} {cnt:>5}x  in {doc_count:>4} docs ({doc_count/len(df)*100:.1f}%)")
