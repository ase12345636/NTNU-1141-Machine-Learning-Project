#!/usr/bin/env python3
"""
Mark any QA json with empty or missing 'answer' as 'unknown'.
Writes updates in-place and prints a summary.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA_DIR = ROOT / 'output_nih' / 'qa'

updated_files = []
checked = 0
for jf in QA_DIR.rglob('*.json'):
    try:
        obj = json.loads(jf.read_text())
    except Exception:
        continue
    checked += 1
    ans = obj.get('answer')
    if ans is None or (isinstance(ans, str) and ans.strip() == ''):
        obj['answer'] = 'unknown'
        jf.write_text(json.dumps(obj, ensure_ascii=False))
        updated_files.append(str(jf.relative_to(ROOT)))

print(f'Checked {checked} files, updated {len(updated_files)} files.')
# print a few examples
for p in updated_files[:50]:
    print(p)
