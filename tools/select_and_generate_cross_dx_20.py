from pathlib import Path
import json
import random
import sys

BASE = Path('d:/CXReasonBench')
OUT = BASE / 'output_nih'
PNT = OUT / 'pnt_on_cxr'
SEGM = OUT / 'segmask_bodypart'

def list_dicoms_in_dir(d):
    if not d.exists():
        return []
    files = []
    for p in d.rglob('*'):
        if p.is_file() and p.suffix.lower() in ('.json','.txt','.csv'):
            files.append(p.stem)
    return sorted(set(files))

def select_dicoms_per_dx(dx_dirs, per_dx=4, total=20):
    selected = []
    dx_list = sorted(dx_dirs)
    i = 0
    while len(selected) < total and i < len(dx_list):
        dx = dx_list[i]
        dicoms = dx_dirs[dx]
        if not dicoms:
            i += 1
            continue
        take = min(per_dx, len(dicoms), total - len(selected))
        chosen = random.sample(dicoms, take)
        selected.append({'dx': dx, 'dicoms': chosen})
        i += 1
    return selected

def write_dx_by_dicoms(sel, out_file):
    # format: list of {dx: <dx>, dicoms: [..]}
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(sel, f, indent=2, ensure_ascii=False)

def make_minimal_qa(dx, dicom_id):
    # create minimal path1/path2/re-path1 structure with basic/<dicom>.json
    qa_root = OUT / 'qa' / dx
    # path1 init
    p_path1 = qa_root / 'path1' / 'init' / 'basic'
    p_path1.mkdir(parents=True, exist_ok=True)
    p_path1_file = p_path1 / f"{dicom_id}.json"
    content = {
        'img_path': f'images/{dicom_id}.png',
        'question': 'Is the image includable?',
        'answer': 'yes',
        'demographics': {}
    }
    with open(p_path1_file, 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    # path2 stages: stage1, stage2, stage3
    for stage in ('stage1','stage2','stage3'):
        p = qa_root / 'path2' / stage / 'basic'
        p.mkdir(parents=True, exist_ok=True)
        with open(p / f"{dicom_id}.json", 'w', encoding='utf-8') as f:
            json.dump({'img_path': f'images/{dicom_id}.png', 'answer': 'stub'}, f, indent=2, ensure_ascii=False)

    # re-path1: init + stage1..stage4
    p = qa_root / 're-path1' / 'init' / 'basic'
    p.mkdir(parents=True, exist_ok=True)
    with open(p / f"{dicom_id}.json", 'w', encoding='utf-8') as f:
        json.dump({'img_path': f'images/{dicom_id}.png', 'note': 'review init'}, f, indent=2, ensure_ascii=False)
    for stage in ('stage1','stage2','stage3','stage4'):
        p = qa_root / 're-path1' / stage / 'basic'
        p.mkdir(parents=True, exist_ok=True)
        with open(p / f"{dicom_id}.json", 'w', encoding='utf-8') as f:
            json.dump({'img_path': f'images/{dicom_id}.png', 'review': 'stub'}, f, indent=2, ensure_ascii=False)

def main():
    random.seed(42)
    dx_dirs = {}
    # gather dx folders from pnt_on_cxr and segmask_bodypart
    for root in (PNT, SEGM):
        if not root.exists():
            continue
        for dx in sorted([d.name for d in root.iterdir() if d.is_dir()]):
            path = root / dx
            dicoms = list_dicoms_in_dir(path)
            if dx not in dx_dirs:
                dx_dirs[dx] = []
            dx_dirs[dx].extend(dicoms)
    # dedupe lists
    for k in dx_dirs:
        dx_dirs[k] = sorted(set(dx_dirs[k]))

    if not dx_dirs:
        print('No dx folders with dicoms found under pnt_on_cxr or segmask_bodypart')
        sys.exit(1)

    selected = select_dicoms_per_dx(dx_dirs, per_dx=4, total=20)
    out_file = OUT / 'dx_by_dicoms_cross_dx_20.json'
    write_dx_by_dicoms(selected, out_file)
    print(f'Wrote {out_file} with {sum(len(x["dicoms"]) for x in selected)} dicoms across {len(selected)} dxs')

    # generate minimal QA stubs
    for item in selected:
        dx = item['dx']
        for d in item['dicoms']:
            make_minimal_qa(dx, d)

    print('Generated minimal QA stubs under output_nih/qa')

if __name__ == '__main__':
    main()
