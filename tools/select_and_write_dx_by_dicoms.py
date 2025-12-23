import os
import json

BASE = r"d:\CXReasonBench\output_nih\pnt_on_cxr"
OUT = r"d:\CXReasonBench\output_nih\dx_by_dicoms_crossdx_20.json"
# dx -> desired counts
DX_COUNTS = {
    "cardiomegaly": 4,
    "inclusion": 4,
    "carina_angle": 4,
    "inspiration": 4,
    "mediastinal_widening": 4,
}

out = {}
for dx, cnt in DX_COUNTS.items():
    dx_dir = os.path.join(BASE, dx)
    if not os.path.isdir(dx_dir):
        print(f"Warning: dx folder not found: {dx_dir}")
        out[dx] = []
        continue
    files = [f for f in os.listdir(dx_dir) if f.lower().endswith('.png') or f.lower().endswith('.jpg')]
    files.sort()
    selected = [os.path.splitext(f)[0] for f in files[:cnt]]
    out[dx] = selected

with open(OUT, 'w', encoding='utf-8') as fh:
    json.dump(out, fh, indent=2)

print(f'Wrote {OUT} with dx counts: {DX_COUNTS}')
