#!/usr/bin/env python3
"""
Generate QA stubs only for dicoms that have existing point/segmask files.
For each dx found under output_nih/pnt_on_cxr or output_nih/segmask_bodypart,
collect up to 20 dicom ids that map to existing image files, write QA stubs
(with img_path filled) and produce dx_by_dicoms_real_20.json.
"""
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NIH = ROOT / 'output_nih'
PNT_DIR = OUTPUT_NIH / 'pnt_on_cxr'
SEGMASK_DIR = OUTPUT_NIH / 'segmask_bodypart'
QA_DIR = OUTPUT_NIH / 'qa'
DATASET_DIRS = [ROOT / 'dataset', ROOT / 'CXAS']
OUT_FILE = OUTPUT_NIH / 'dx_by_dicoms_real_20.json'
SAMPLE_PER_DX = 20
IMAGE_INDEX = OUTPUT_NIH / 'image_index.json'


def find_image_for_id(dicom_id):
    # Fast path: use precomputed image index if available
    if IMAGE_INDEX.exists():
        try:
            import json
            idx = json.load(open(IMAGE_INDEX))
            if dicom_id in idx:
                return idx[dicom_id]
        except Exception:
            pass
    for root in DATASET_DIRS:
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if p.is_file() and p.stem.startswith(dicom_id):
                return p.resolve().as_posix()
    return None


def collect_ids_from_sources(dx):
    """Generator: yield candidate dicom ids one-by-one from pnt_on_cxr and segmask_bodypart.

    This avoids building a large list in memory and processes each id immediately.
    """
    seen = set()
    # pnt_on_cxr: images are directly under the dx folder
    p = PNT_DIR / dx
    if p.exists():
        for f in p.iterdir():
            if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                stem = f.stem
                if stem not in seen:
                    seen.add(stem)
                    yield stem
    # segmask_bodypart: subfolders (bodyparts) contain images
    p2 = SEGMASK_DIR / dx
    if p2.exists():
        for sub in p2.iterdir():
            if not sub.is_dir():
                continue
            for f in sub.iterdir():
                if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    stem = f.stem
                    if stem not in seen:
                        seen.add(stem)
                        yield stem


def ensure_dirs(dx):
    (QA_DIR / dx / 'path1' / 'init' / 'basic').mkdir(parents=True, exist_ok=True)
    for s in ('stage1','stage2','stage3'):
        (QA_DIR / dx / 'path2' / s / 'basic').mkdir(parents=True, exist_ok=True)
    (QA_DIR / dx / 're-path1' / 'init' / 'basic').mkdir(parents=True, exist_ok=True)
    for s in ('stage1','stage2','stage3','stage4'):
        (QA_DIR / dx / 're-path1' / s / 'basic').mkdir(parents=True, exist_ok=True)


def write_stub(dx, dicom_id, img_path):
    qpath = QA_DIR / dx
    # path1
    p = qpath / 'path1' / 'init' / 'basic' / f'{dicom_id}.json'
    p.write_text(json.dumps({'img_path': img_path, 'question': f'Is there evidence related to {dx}?', 'answer': '', 'dicom_id': dicom_id}, ensure_ascii=False))
    # path2
    for stage, text in [('stage1','Identify bodypart'), ('stage2','Measurement/score'), ('stage3','Final decision')]:
        p = qpath / 'path2' / stage / 'basic' / f'{dicom_id}.json'
        p.write_text(json.dumps({'img_path': img_path, 'question': text, 'answer': '', 'dicom_id': dicom_id}, ensure_ascii=False))
    # re-path1
    p = qpath / 're-path1' / 'init' / 'basic' / f'{dicom_id}.json'
    p.write_text(json.dumps({'img_path': img_path, 'question': 'Review: init', 'answer': '', 'dicom_id': dicom_id}, ensure_ascii=False))
    for i in range(1,5):
        p = qpath / 're-path1' / f'stage{i}' / 'basic' / f'{dicom_id}.json'
        p.write_text(json.dumps({'img_path': img_path, 'question': f'Review stage {i}', 'answer': '', 'dicom_id': dicom_id}, ensure_ascii=False))


def main():
    # Only consider diseases that have point-on-cxr data (PNT_DIR).
    # If a disease lacks a pnt_on_cxr folder, skip it.
    if not PNT_DIR.exists():
        print('No pnt_on_cxr directory found. Exiting.')
        return
    pnt_dxs = [p.name for p in PNT_DIR.iterdir() if p.is_dir()]
    if not pnt_dxs:
        print('No disease folders under pnt_on_cxr. Exiting.')
        return
    dxs = sorted(pnt_dxs)
    result = {}
    for dx in dxs:
        candidates = collect_ids_from_sources(dx)
        valid = []
        ensure_dirs(dx)
        for cid in candidates:
            if len(valid) >= SAMPLE_PER_DX:
                break
            img = find_image_for_id(cid)
            if img:
                write_stub(dx, cid, img)
                valid.append(cid)
        if not valid:
            print(f'Skipping {dx}: no valid images found from pnt/segmask sources')
            continue
        result[dx] = valid
        print(f'{dx}: generated {len(valid)} stubs')
    OUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print('Wrote', OUT_FILE)

if __name__ == '__main__':
    main()
