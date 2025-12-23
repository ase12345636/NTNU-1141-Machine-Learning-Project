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
    p.add_argument('--out', default='dx_by_dicoms_complete_sampled.json', help='Output filtered json file')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def has_all_required_stages(qa_base_dir, dx, dicom):
    base = os.path.join(qa_base_dir, dx, 'path1')
    # init
    init_path = os.path.join(base, 'init', 'basic', f'{dicom}.json')
    if not os.path.exists(init_path):
        return False

    # criteria / stage1
    criteria_found = False
    for variant in ['stage1', 'criteria']:
        files = glob(os.path.join(base, variant, '*', f'{dicom}.json'))
        if files:
            criteria_found = True
            break
    if not criteria_found:
        return False

    # bodypart / stage2 (allow any variant)
    bodypart_found = False
    for variant in ['stage2', 'bodypart']:
        files = glob(os.path.join(base, variant, '*', f'{dicom}.json'))
        if files:
            bodypart_found = True
            break
    if not bodypart_found:
        return False

    # measurement / stage3
    measurement_found = False
    for variant in ['stage3', 'measurement']:
        path = os.path.join(base, variant, 'basic', f'{dicom}.json')
        if os.path.exists(path):
            measurement_found = True
            break
    if not measurement_found:
        return False

    # final / stage4
    final_found = False
    for variant in ['stage4', 'final']:
        path = os.path.join(base, variant, 'basic', f'{dicom}.json')
        if os.path.exists(path):
            final_found = True
            break
    if not final_found:
        return False

    return True


def main():
    args = config()
    random.seed(args.seed)

    with open(args.dx_by_dicoms_file, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    dicom_list = data.get(args.dx, [])
    print(f'Total dicoms in dx_by_dicoms for {args.dx}: {len(dicom_list)}')

    # Build sets of available dicom ids per stage by scanning the QA folders once each.
    base = os.path.join(args.qa_base_dir, args.dx, 'path1')

    def collect_ids_under(subpath):
        p = os.path.join(base, subpath) if subpath else base
        ids = set()
        if not os.path.exists(p):
            return ids
        for root, _, files in os.walk(p):
            for fn in files:
                if fn.lower().endswith('.json'):
                    ids.add(os.path.splitext(fn)[0])
        return ids

    init_ids = collect_ids_under(os.path.join('init', 'basic'))
    criteria_ids = collect_ids_under('stage1') | collect_ids_under('criteria')
    bodypart_ids = collect_ids_under('stage2') | collect_ids_under('bodypart')
    measurement_ids = collect_ids_under(os.path.join('stage3', 'basic')) | collect_ids_under('measurement')
    final_ids = collect_ids_under(os.path.join('stage4', 'basic')) | collect_ids_under('final')

    print(f'Found init: {len(init_ids)}, criteria: {len(criteria_ids)}, bodypart: {len(bodypart_ids)}, measurement: {len(measurement_ids)}, final: {len(final_ids)}')

    # Compute intersection of available ids across required stages, restricted to the provided dicom list
    provided_set = set(dicom_list)
    complete_set = init_ids & criteria_ids & bodypart_ids & measurement_ids & final_ids & provided_set
    complete = sorted(list(complete_set))

    print(f'Dicoms with complete stages for {args.dx}: {len(complete)}')

    if not complete:
        print('No dicoms with complete QA stages found. Exiting.')
        return

    sampled = complete if len(complete) <= args.sample_size else random.sample(complete, args.sample_size)

    out_data = {args.dx: sampled}
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(out_data, fh, indent=2)

    print(f'Kept {len(sampled)} dicoms for dx={args.dx} -> {args.out}')


if __name__ == '__main__':
    main()
