#!/usr/bin/env python3
"""
Build a persistent image index (basename -> list of full paths) for pnt_on_cxr and segmask_bodypart
and write to `output_nih/image_index.json`.

Usage:
    python tools/build_image_index.py --output /mnt/d/CXReasonBench/output_nih
"""
import os
import json
import argparse


def build_index(root):
    idx = {}
    pnt = os.path.join(root, 'pnt_on_cxr')
    seg = os.path.join(root, 'segmask_bodypart')

    def add(path):
        name = os.path.splitext(os.path.basename(path))[0]
        idx.setdefault(name, []).append(path)

    if os.path.isdir(pnt):
        for dx in os.listdir(pnt):
            dxp = os.path.join(pnt, dx)
            if not os.path.isdir(dxp):
                continue
            for fn in os.listdir(dxp):
                if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                    add(os.path.join(dxp, fn))

    if os.path.isdir(seg):
        for dx in os.listdir(seg):
            dxp = os.path.join(seg, dx)
            if not os.path.isdir(dxp):
                continue
            for bp in os.listdir(dxp):
                bpp = os.path.join(dxp, bp)
                if not os.path.isdir(bpp):
                    continue
                for fn in os.listdir(bpp):
                    if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                        add(os.path.join(bpp, fn))

    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='Path to output_nih root')
    args = parser.parse_args()
    root = args.output
    if not os.path.isdir(root):
        print('Invalid output root:', root)
        return
    idx = build_index(root)
    out_path = os.path.join(root, 'image_index.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)
    print('Wrote', out_path, 'with', len(idx), 'keys')


if __name__ == '__main__':
    main()
