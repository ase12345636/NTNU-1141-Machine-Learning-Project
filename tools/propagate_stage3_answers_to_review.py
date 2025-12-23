#!/usr/bin/env python3
"""
Propagate answers from path2/stage3/basic/<id>.json into re-path1/*/basic/<id>.json when the latter's 'answer' is empty.
Usage: python tools/propagate_stage3_answers_to_review.py
Writes updates in-place and prints a summary.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA_DIR = ROOT / 'output_nih' / 'qa'

summary = {}
for dx in sorted([p.name for p in QA_DIR.iterdir() if p.is_dir()]):
    dx_dir = QA_DIR / dx
    stage3_dir = dx_dir / 'path2' / 'stage3' / 'basic'
    if not stage3_dir.exists():
        continue
    # read all stage3 answers
    stage3_answers = {}
    for f in stage3_dir.glob('*.json'):
        try:
            obj = json.loads(f.read_text())
            aid = obj.get('dicom_id') or f.stem
            stage3_answers[aid] = obj.get('answer')
        except Exception:
            continue
    updated = 0
    checked = 0
    # iterate re-path1 stages
    for stage in ['init','stage1','stage2','stage3','stage4']:
        pdir = dx_dir / 're-path1' / stage / 'basic'
        if not pdir.exists():
            continue
        for jf in pdir.glob('*.json'):
            checked += 1
            try:
                obj = json.loads(jf.read_text())
            except Exception:
                continue
            if obj.get('answer'):
                continue
            aid = obj.get('dicom_id') or jf.stem
            new_ans = stage3_answers.get(aid)
            if new_ans:
                obj['answer'] = new_ans
                jf.write_text(json.dumps(obj, ensure_ascii=False))
                updated += 1
    summary[dx] = {'checked': checked, 'updated': updated}

print('Propagation summary:')
for dx, v in summary.items():
    print(f"{dx}: checked={v['checked']} updated={v['updated']}")
