#!/usr/bin/env python3
"""
Ensure each disease has 20 QA stubs and write dx_by_dicoms_cross_all_dx_20_each.json
Usage: python tools/ensure_20_per_dx_and_prepare_eval.py
"""
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NIH = ROOT / 'output_nih'
QA_DIR = OUTPUT_NIH / 'qa'
NIH_CSV_DIR = ROOT / 'nih_cxr14'
OUT_FILE = OUTPUT_NIH / 'dx_by_dicoms_cross_all_dx_20_each.json'
SAMPLE_PER_DX = 20


def list_dxs_from_nih():
    dxs = []
    if NIH_CSV_DIR.exists():
        for f in sorted(NIH_CSV_DIR.glob('*.csv')):
            dxs.append(f.stem)
    # fallback: use existing qa dirs
    if not dxs and QA_DIR.exists():
        for p in sorted(QA_DIR.iterdir()):
            if p.is_dir():
                dxs.append(p.name)
    return dxs


def collect_existing_ids(dx):
    ids = set()
    base = QA_DIR / dx / 'path1' / 'init' / 'basic'
    if base.exists():
        for f in base.glob('*.json'):
            ids.add(f.stem)
    return sorted(ids)


def find_image_file_for_id(dicom_id):
    # Search common dataset folders for a file that starts with the dicom id
    # Return absolute POSIX path string or None
    # Limit search to these candidate roots for performance
    candidate_roots = [ROOT / 'dataset', ROOT / 'CXAS']
    for root in candidate_roots:
        if not root.exists():
            continue
        # search for files whose stem startswith dicom_id
        for p in root.rglob('*'):
            if p.is_file():
                try:
                    if p.stem.startswith(dicom_id):
                        return p.resolve().as_posix()
                except Exception:
                    continue
    return None


def ensure_dirs_for_dx(dx):
    # create expected structure
    (QA_DIR / dx / 'path1' / 'init' / 'basic').mkdir(parents=True, exist_ok=True)
    for s in ('stage1','stage2','stage3'):
        (QA_DIR / dx / 'path2' / s / 'basic').mkdir(parents=True, exist_ok=True)
    (QA_DIR / dx / 're-path1' / 'init' / 'basic').mkdir(parents=True, exist_ok=True)
    for s in ('stage1','stage2','stage3','stage4'):
        (QA_DIR / dx / 're-path1' / s / 'basic').mkdir(parents=True, exist_ok=True)


def write_minimal(dx, dicom_id):
    qpath = QA_DIR / dx
    # find image file for this dicom id; if not found, return False to indicate skip
    img = find_image_file_for_id(dicom_id)
    if img is None:
        return False
    # path1
    p = qpath / 'path1' / 'init' / 'basic' / f'{dicom_id}.json'
    p.write_text(json.dumps({'img_path': img, 'question':f'Is there evidence related to {dx}?','answer':'','dicom_id':dicom_id}, ensure_ascii=False))
    # path2
    for stage, text in [('stage1','Identify bodypart'), ('stage2','Measurement/score'), ('stage3','Final decision')]:
        p = qpath / 'path2' / stage / 'basic' / f'{dicom_id}.json'
        p.write_text(json.dumps({'img_path': img, 'question':text,'answer':'','dicom_id':dicom_id}, ensure_ascii=False))
    # re-path1
    p = qpath / 're-path1' / 'init' / 'basic' / f'{dicom_id}.json'
    p.write_text(json.dumps({'img_path': img, 'question':'Review: init','answer':'','dicom_id':dicom_id}, ensure_ascii=False))
    for i in range(1,5):
        p = qpath / 're-path1' / f'stage{i}' / 'basic' / f'{dicom_id}.json'
        p.write_text(json.dumps({'img_path': img, 'question':f'Review stage {i}','answer':'','dicom_id':dicom_id}, ensure_ascii=False))
    return True


def main():
    dxs = list_dxs_from_nih()
    if not dxs:
        print('No dx list found (nih_cxr14 empty and no qa dirs). Exiting.')
        return
    result = {}
    for dx in dxs:
        ensure_dirs_for_dx(dx)
        existing = collect_existing_ids(dx)
        needed = SAMPLE_PER_DX - len(existing)
        ids = existing.copy()
        # generate synthetic ids if needed
        idx = 0
        # prefer existing/qas from source lists; if not enough, try to use ids from pnt/segmask
        # collect candidate ids from pnt_on_cxr and segmask_bodypart if available
        # NOTE: we already ensured dirs; now try to fill up to SAMPLE_PER_DX using any available ids
        # first, try existing ids (already in 'existing')
        # if still short, attempt to use ids discovered from pnt_on_cxr/segmask (from previously generated list)
        # finally, attempt synthetic ids but only if an image exists for them (find_image_file_for_id)
        while len(ids) < SAMPLE_PER_DX:
            candidate = f'syn_{dx}_{idx:04d}'
            if candidate in ids:
                idx += 1
                continue
            ok = write_minimal(dx, candidate)
            if not ok:
                # could not find image for synthetic id; increment and continue
                idx += 1
                # to avoid infinite loops, after many tries break
                if idx > 1000:
                    break
                continue
            ids.append(candidate)
            idx += 1
        # if more than SAMPLE_PER_DX, trim randomly
        if len(ids) > SAMPLE_PER_DX:
            ids = random.sample(ids, SAMPLE_PER_DX)
        result[dx] = ids
        print(f'{dx}: {len(ids)} entries')
    OUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print('Wrote', OUT_FILE)

if __name__ == '__main__':
    main()
