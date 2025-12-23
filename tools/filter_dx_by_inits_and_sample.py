import os
import json
import argparse
import random


def config():
    p = argparse.ArgumentParser()
    p.add_argument('--dx_by_dicoms_file', required=True, help='Path to dx_by_dicoms.json')
    p.add_argument('--qa_base_dir', required=True, help='Root QA directory (e.g. output_nih/qa)')
    p.add_argument('--dx', default='inclusion', help='Diagnostic task to keep')
    p.add_argument('--sample_size', type=int, default=20, help='Number of dicoms to sample')
    p.add_argument('--out', default='dx_by_dicoms_sampled.json', help='Output filtered json file')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = config()
    random.seed(args.seed)

    # Load dx_by_dicoms
    with open(args.dx_by_dicoms_file, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    dicom_list = data.get(args.dx, [])

    # Fast discovery: scan the QA init/basic folder for available init json filenames
    init_dir = os.path.join(args.qa_base_dir, args.dx, 'path1', 'init', 'basic')
    if not os.path.isdir(init_dir):
        print(f'Warning: init directory not found: {init_dir}')

    available_inits = set()
    # If directory exists, walk it (should be small per-dx) and collect basenames
    if os.path.isdir(init_dir):
        for root, _, files in os.walk(init_dir):
            for fn in files:
                if fn.lower().endswith('.json'):
                    available_inits.add(os.path.splitext(fn)[0])

    # Filter the provided dicoms by presence in available_inits
    available = [d for d in dicom_list if d in available_inits]

    print(f'Total dicoms in dx_by_dicoms for {args.dx}: {len(dicom_list)}')
    print(f'Init jsons found under {init_dir}: {len(available_inits)}')
    print(f'Candidate dicoms with init present: {len(available)}')

    if not available:
        print(f'No available init files found for dx={args.dx} under {args.qa_base_dir}')
        return

    sampled = available if len(available) <= args.sample_size else random.sample(available, args.sample_size)

    out_data = {args.dx: sampled}
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(out_data, fh, indent=2)

    print(f'Kept {len(sampled)} dicoms for dx={args.dx} -> {args.out}')


if __name__ == '__main__':
    main()
