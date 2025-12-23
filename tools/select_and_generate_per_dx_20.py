#!/usr/bin/env python3
"""
Select up to 20 dicoms per disease (dx) from available point/segmask outputs
and generate minimal QA stubs for each selected dicom so the evaluation pipeline
can run end-to-end per-disease.

Usage: python tools/select_and_generate_per_dx_20.py
Generates: output_nih/dx_by_dicoms_cross_all_dx_20_each.json
and QA files under output_nih/qa/<dx>/...
"""
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NIH = ROOT / "output_nih"
PNT_DIR = OUTPUT_NIH / "pnt_on_cxr"
SEGMASK_DIR = OUTPUT_NIH / "segmask_bodypart"
QA_DIR = OUTPUT_NIH / "qa"

SAMPLE_PER_DX = 20

def list_dx_sources():
    dxs = set()
    if PNT_DIR.exists():
        for p in PNT_DIR.iterdir():
            if p.is_dir():
                dxs.add(p.name)
    if SEGMASK_DIR.exists():
        for p in SEGMASK_DIR.iterdir():
            if p.is_dir():
                dxs.add(p.name)
    return sorted(dxs)

def collect_dicoms_for_dx(dx):
    ids = set()
    p1 = PNT_DIR / dx
    if p1.exists():
        for f in p1.rglob("*"):
            if f.is_file():
                ids.add(f.stem)
    p2 = SEGMASK_DIR / dx
    if p2.exists():
        for f in p2.rglob("*"):
            if f.is_file():
                ids.add(f.stem)
    return sorted(ids)

def ensure_dirs_for_dx(dx):
    # path1
    (QA_DIR / dx / 'path1' / 'init' / 'basic').mkdir(parents=True, exist_ok=True)
    # path2 stages
    for s in ('stage1','stage2','stage3'):
        (QA_DIR / dx / 'path2' / s / 'basic').mkdir(parents=True, exist_ok=True)
    # re-path1 stages
    (QA_DIR / dx / 're-path1' / 'init' / 'basic').mkdir(parents=True, exist_ok=True)
    for s in ('stage1','stage2','stage3','stage4'):
        (QA_DIR / dx / 're-path1' / s / 'basic').mkdir(parents=True, exist_ok=True)

def make_minimal_qa(dx, dicom_id):
    # simple templates; fields can be extended later
    q_path = QA_DIR / dx
    # path1/init
    path1 = q_path / 'path1' / 'init' / 'basic' / f"{dicom_id}.json"
    data1 = {
        'img_path': '',
        'question': f'Is there evidence related to {dx}?',
        'answer': '',
        'dicom_id': dicom_id,
    }
    path1.write_text(json.dumps(data1, ensure_ascii=False))
    # path2 stages: bodypart, measurement, final
    for stage, text in [('stage1','Identify bodypart'), ('stage2','Measurement/score'), ('stage3','Final decision')]:
        p = q_path / 'path2' / stage / 'basic' / f"{dicom_id}.json"
        obj = {'img_path':'', 'question': text, 'answer':'', 'dicom_id': dicom_id}
        p.write_text(json.dumps(obj, ensure_ascii=False))
    # re-path1: review workflow
    p = q_path / 're-path1' / 'init' / 'basic' / f"{dicom_id}.json"
    p.write_text(json.dumps({'img_path':'','question':'Review: init','answer':'','dicom_id':dicom_id}, ensure_ascii=False))
    for i in range(1,5):
        p = q_path / 're-path1' / f'stage{i}' / 'basic' / f"{dicom_id}.json"
        p.write_text(json.dumps({'img_path':'','question':f'Review stage {i}','answer':'','dicom_id':dicom_id}, ensure_ascii=False))

def main():
    dxs = list_dx_sources()
    if not dxs:
        print('No dx folders found under pnt_on_cxr or segmask_bodypart. Exiting.')
        return
    result = {}
    for dx in dxs:
        dicoms = collect_dicoms_for_dx(dx)
        if not dicoms:
            print(f'No dicoms found for {dx}, skipping.')
            continue
        # sample up to SAMPLE_PER_DX
        if len(dicoms) <= SAMPLE_PER_DX:
            sampled = dicoms
        else:
            sampled = random.sample(dicoms, SAMPLE_PER_DX)
        ensure_dirs_for_dx(dx)
        for d in sampled:
            make_minimal_qa(dx, d)
        result[dx] = sampled
        print(f'Generated {len(sampled)} QA stubs for {dx}')

    out = OUTPUT_NIH / 'dx_by_dicoms_cross_all_dx_20_each.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print('Wrote', out)

if __name__ == '__main__':
    main()
