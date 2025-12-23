import os
import json
import argparse
import random
from glob import glob


def config():
    p = argparse.ArgumentParser()
    p.add_argument('--dx_by_dicoms_file', required=True, help='Path to dx_by_dicoms.json')
    p.add_argument('--qa_base_dir', required=True, help='Root QA directory (e.g. output_nih/qa)')
    p.add_argument('--dx', default='inclusion', help='Diagnostic task to keep')
    p.add_argument('--sample_size', type=int, default=20, help='Number of dicoms to sample')
    p.add_argument('--out', default='dx_by_dicoms_path2_repath1_sampled.json', help='Output filtered json file')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


HEADER_STAGE = ['guidance-bodypart', 'guidance-measurement', 'guidance-final',
                'review-init', 'review-criteria', 'review-bodypart', 'review-measurement', 'review-final']


def collect_stage_dirs(qa_base_dir, dx):
    path2_base = os.path.join(qa_base_dir, dx, 'path2')
    repath1_base = os.path.join(qa_base_dir, dx, 're-path1')

    path2_dirs = sorted([p for p in glob(os.path.join(path2_base, '*')) if os.path.isdir(p)])
    repath1_dirs = sorted([p for p in glob(os.path.join(repath1_base, '*')) if os.path.isdir(p)])

    return path2_dirs, repath1_dirs


def has_all_path_files(stage_dirs, dicom):
    # For each stage dir we expect a basic/<dicom>.json (matching evaluate_guidance expectations)
    for sd in stage_dirs:
        candidate = os.path.join(sd, 'basic', f'{dicom}.json')
        if not os.path.exists(candidate):
            return False
    return True


def main():
    args = config()
    random.seed(args.seed)

    with open(args.dx_by_dicoms_file, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    provided = data.get(args.dx, [])
    print(f'Total dicoms provided for {args.dx}: {len(provided)}')

    path2_dirs, repath1_dirs = collect_stage_dirs(args.qa_base_dir, args.dx)
    possible_stage_lst = path2_dirs + repath1_dirs

    print(f'Found {len(path2_dirs)} path2 dirs and {len(repath1_dirs)} re-path1 dirs (total {len(possible_stage_lst)})')

    if len(possible_stage_lst) != len(HEADER_STAGE):
        print(f'Warning: header_stage length ({len(HEADER_STAGE)}) != discovered stage dirs ({len(possible_stage_lst)}).')
        print('This means the QA folder layout for this dx does not match the original CXReasonBench expectation for guidance. Exiting without sampling.')
        return

    # Check each provided dicom has a basic json in every stage dir
    eligible = []
    for dicom in provided:
        if has_all_path_files(possible_stage_lst, dicom):
            eligible.append(dicom)

    print(f'Eligible dicoms with complete path2+re-path1 files: {len(eligible)}')

    if not eligible:
        print('No eligible dicoms found. Exiting.')
        return

    sampled = eligible if len(eligible) <= args.sample_size else random.sample(eligible, args.sample_size)

    out_data = {args.dx: sampled}
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(out_data, fh, indent=2)

    print(f'Wrote {len(sampled)} dicoms to {args.out}')


if __name__ == '__main__':
    main()
