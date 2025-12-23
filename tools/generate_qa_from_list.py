#!/usr/bin/env python3
"""
Generate full QA stubs for dicoms listed in a dx_by_dicoms JSON file.
Usage: python tools/generate_qa_from_list.py /mnt/d/CXReasonBench/output_nih/dx_by_dicoms_real_20.json
Writes QA files under output_nih/qa/<dx>/... and a missing list at output_nih/missing_images.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NIH = ROOT / 'output_nih'
QA_DIR = OUTPUT_NIH / 'qa'
DATASET_DIRS = [ROOT / 'dataset' / 'images_001' / 'images', ROOT / 'dataset']


def find_image(dicom_id):
    # check pnt_on_cxr first
    pnt = OUTPUT_NIH / 'pnt_on_cxr'
    if pnt.exists():
        for dxdir in pnt.iterdir():
            if not dxdir.is_dir():
                continue
            candidate = dxdir / f'{dicom_id}.png'
            if candidate.exists():
                return candidate.resolve().as_posix()
    # search dataset folders fast (non-recursive common images folder)
    for root in DATASET_DIRS:
        if not root.exists():
            continue
        # images may be in nested 'images' directory
        for p in root.glob('**/*'):
            if p.is_file() and p.stem == dicom_id:
                return p.resolve().as_posix()
    return None


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
    if len(sys.argv) < 2:
        print('Usage: python tools/generate_qa_from_list.py <dx_by_dicoms.json>')
        sys.exit(1)
    infile = Path(sys.argv[1])
    if not infile.exists():
        print('Input file not found:', infile)
        sys.exit(1)
    data = json.load(open(infile,'r'))
    missing = {}
    for dx, ids in data.items():
        ensure_dirs(dx)
        missing_dx = []
        created = 0
        for dicom_id in ids:
            if created >= 20:
                break
            img = find_image(dicom_id)
            if not img:
                missing_dx.append(dicom_id)
                continue
            write_stub(dx, dicom_id, img)
            created += 1
        if missing_dx:
            missing[dx] = missing_dx
        print(f'{dx}: created {created} QA stubs, missing {len(missing_dx)}')
    out_missing = OUTPUT_NIH / 'missing_images.json'
    out_missing.write_text(json.dumps(missing, ensure_ascii=False, indent=2))
    print('Wrote missing list to', out_missing)

if __name__ == '__main__':
    main()
