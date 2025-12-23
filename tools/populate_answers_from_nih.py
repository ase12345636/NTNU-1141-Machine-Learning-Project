#!/usr/bin/env python3
"""
Populate QA 'answer' fields using NIH CSVs.
Priority:
 1) use nih_cxr14/<dx>.csv (if exists) — treat listed ImageIndex as positive
 2) fallback to dataset/Data_Entry_2017.csv — match Image Index and disease column

Usage: python tools/populate_answers_from_nih.py
Writes: updates QA json files in place; prints summary.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NIH = ROOT / 'output_nih'
NIH_DIR = ROOT / 'nih_cxr14'
DATA_ENTRY = ROOT / 'dataset' / 'Data_Entry_2017.csv'
QA_DIR = OUTPUT_NIH / 'qa'


def load_nih_dx_csv(dx):
    f = NIH_DIR / f"{dx}.csv"
    if not f.exists():
        return None
    ids = set()
    with open(f, newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            # assume first column contains image index or id
            val = row[0].strip()
            # strip extension if present
            if val.lower().endswith('.png'):
                val = Path(val).stem
            ids.add(val)
    return ids


def load_data_entry():
    if not DATA_ENTRY.exists():
        return None, None
    with open(DATA_ENTRY, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    # create map from image stem to row dict
    m = {}
    for r in rows:
        img = r.get('Image Index') or r.get('ImageIndex') or r.get('Image')
        if not img:
            # try first column
            keys = list(r.keys())
            img = r[keys[0]]
        stem = Path(img).stem
        m[stem] = r
    return m, rows


def update_answers():
    data_entry_map, _ = load_data_entry()
    dxs = [p.name for p in QA_DIR.iterdir() if p.is_dir()]
    summary = {}
    for dx in dxs:
        nih_ids = load_nih_dx_csv(dx)
        files_updated = 0
        files_checked = 0
        dx_dir = QA_DIR / dx
        # iterate all basic json files under path1/path2/re-path1
        for stage_dir in ['path1/init/basic','path2/stage1/basic','path2/stage2/basic','path2/stage3/basic','re-path1/init/basic','re-path1/stage1/basic','re-path1/stage2/basic','re-path1/stage3/basic','re-path1/stage4/basic']:
            pdir = dx_dir / Path(stage_dir)
            if not pdir.exists():
                continue
            for jf in pdir.glob('*.json'):
                files_checked += 1
                try:
                    obj = json.loads(jf.read_text())
                except Exception:
                    continue
                dicom = obj.get('dicom_id') or obj.get('dicom') or Path(jf).stem
                # decide answer
                ans = None
                # Only populate answers for questions that expect a binary Yes/No disease label.
                # Heuristic: question text contains both 'yes' and 'no' options, or starts with 'does',
                # or explicitly mentions 'does this'. Otherwise skip (e.g., 'Identify bodypart').
                qtext = (obj.get('question') or '').lower()
                is_binary_q = False
                if 'yes' in qtext and 'no' in qtext:
                    is_binary_q = True
                elif qtext.strip().startswith('does') or 'does this' in qtext:
                    is_binary_q = True
                elif 'options:' in qtext and ('yes' in qtext or 'no' in qtext):
                    is_binary_q = True
                if not is_binary_q:
                    # skip non-binary questions
                    continue
                if nih_ids is not None:
                    if dicom in nih_ids:
                        ans = 'positive'
                    else:
                        ans = 'negative'
                elif data_entry_map is not None:
                    row = data_entry_map.get(dicom)
                    if row is None:
                        ans = 'unknown'
                    else:
                        # try several column variants
                        key = None
                        candidates = [dx, dx.replace('_',' '), dx.replace('_',' ').title(), dx.title()]
                        for c in candidates:
                            if c in row:
                                key = c
                                break
                        if key and row.get(key) in ('1','1.0','True','true'):
                            ans = 'positive'
                        elif key:
                            ans = 'negative'
                        else:
                            ans = 'unknown'
                else:
                    ans = 'unknown'
                if obj.get('answer') != ans:
                    obj['answer'] = ans
                    jf.write_text(json.dumps(obj, ensure_ascii=False))
                    files_updated += 1
        summary[dx] = {'checked': files_checked, 'updated': files_updated}
    print('Update summary:')
    for dx, v in summary.items():
        print(f"{dx}: checked={v['checked']} updated={v['updated']}")

if __name__ == '__main__':
    update_answers()
