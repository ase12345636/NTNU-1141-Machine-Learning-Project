#!/usr/bin/env python3
"""
Revert any QA json files whose 'answer' == 'unknown' back to empty string.
Writes updates in-place and prints a summary.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA_DIR = ROOT / 'output_nih' / 'qa'

updated = []
for jf in QA_DIR.rglob('*.json'):
    try:
        obj = json.loads(jf.read_text())
    except Exception:
        continue
    if obj.get('answer') == 'unknown':
        obj['answer'] = ''
        jf.write_text(json.dumps(obj, ensure_ascii=False))
        updated.append(str(jf.relative_to(ROOT)))

print(f'Reverted {len(updated)} files.')
for p in updated[:100]:
    print(p)
