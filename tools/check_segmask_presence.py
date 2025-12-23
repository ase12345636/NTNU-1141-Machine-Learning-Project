"""
Check presence of segmentation mask files for images listed in a dx_by_dicoms JSON.

Usage:
  python3 tools/check_segmask_presence.py --dx_json /mnt/d/CXReasonBench/output_nih/dx_by_dicoms_real_20.json --cxreasonbench_base_dir /mnt/d/CXReasonBench

Outputs a per-dx summary of how many images have segmasks and lists a few missing examples.
"""
import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dx_json', required=True)
    parser.add_argument('--cxreasonbench_base_dir', required=True)
    parser.add_argument('--limit_examples', type=int, default=10)
    args = parser.parse_args()

    segmask_base = os.path.join(args.cxreasonbench_base_dir, 'segmask_bodypart')

    with open(args.dx_json, 'r', encoding='utf-8') as f:
        dx_by_dicoms = json.load(f)

    total_missing = 0
    for dx, dicom_list in dx_by_dicoms.items():
        missing = []
        present_count = 0
        for dicom in dicom_list:
            # look for any segmask under segmask_base/<dx>/*/<dicom>.png
            found = False
            dx_dir = os.path.join(segmask_base, dx)
            if os.path.isdir(dx_dir):
                for root, dirs, files in os.walk(dx_dir):
                    fname = f"{dicom}.png"
                    if fname in files:
                        found = True
                        break
            if found:
                present_count += 1
            else:
                missing.append(dicom)

        total = len(dicom_list)
        total_missing += len(missing)
        print(f"{dx}: {present_count}/{total} present, {len(missing)} missing")
        if missing:
            examples = missing[:args.limit_examples]
            for m in examples:
                print(f"  missing: {m}")

    print(f"Total missing segmasks: {total_missing}")


if __name__ == '__main__':
    main()
