import os
import json
import argparse
import random
from glob import glob

TEMPLATE_PATHS = [
    "/inclusion/lung_both/{dicom}.png",
    "/inclusion/lung_left/{dicom}.png",
    "/inclusion/lung_right/{dicom}.png",
]

def config():
    p = argparse.ArgumentParser()
    p.add_argument('--dx_by_dicoms_file', required=True)
    p.add_argument('--qa_base_dir', required=True)
    p.add_argument('--dx', default='inclusion')
    p.add_argument('--sample_size', type=int, default=20)
    p.add_argument('--out_dx_by_dicoms', required=True)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_json(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def make_init_json(dicom):
    return {
        "img_path": [p.format(dicom=dicom) for p in TEMPLATE_PATHS],
        "question": ["Please confirm the initial view and presence of the target bodyparts."],
        "answer": ["confirmed"]
    }


def make_path2_stage1_json(dicom):
    return {
        "question": ["Which image shows both lungs? Options: (a)1 (b)2 (c)3"],
        "answer": ["(b) 2nd image"],
        "img_path": [p.format(dicom=dicom) for p in TEMPLATE_PATHS]
    }


def make_path2_stage2_json(dicom):
    return {
        "question": "Measure the vertical distance between landmarks and report numeric value in mm.",
        "answer": "10.0",
        "img_path": [p.format(dicom=dicom) for p in TEMPLATE_PATHS]
    }


def make_path2_stage3_json(dicom):
    return {
        "question": "Final diagnosis: is inclusion present? (yes/no)",
        "answer": "yes"
    }


def make_repath1_init_json(dicom):
    return {
        "img_path": [p.format(dicom=dicom) for p in TEMPLATE_PATHS],
        "question": "Review init: is the previous selection appropriate?",
        "answer": "yes"
    }


def make_repath1_stage1_json(dicom):
    return {
        "question": ["Are the criteria A and B met?"],
        "answer": ["yes"]
    }


def make_repath1_stage2_json(dicom):
    # img_path per-question is list-of-lists
    return {
        "question": ["Identify lungs in the provided masks."],
        "answer": ["(b) 2nd image"],
        "img_path": [[p.format(dicom=dicom) for p in TEMPLATE_PATHS]]
    }


def make_repath1_stage3_json(dicom):
    return {
        "question": "Please provide measured value.",
        "answer": "10.0"
    }


def make_repath1_stage4_json(dicom):
    return {
        "question": "Final review diagnosis (yes/no).",
        "answer": "yes"
    }


def main():
    args = config()
    random.seed(args.seed)

    with open(args.dx_by_dicoms_file, 'r', encoding='utf-8') as fh:
        dx_map = json.load(fh)

    dicoms = dx_map.get(args.dx, [])
    if not dicoms:
        print('No dicoms found in dx_by_dicoms_file for', args.dx)
        return

    sampled = dicoms[:args.sample_size] if len(dicoms) >= args.sample_size else dicoms

    # For each sampled dicom, create minimal QA across path1, path2 and re-path1
    for dicom in sampled:
        # path1/init/basic
        p = os.path.join(args.qa_base_dir, args.dx, 'path1', 'init', 'basic')
        ensure_dir(p)
        write_json(os.path.join(p, f"{dicom}.json"), make_init_json(dicom))

        # path2/stage1/basic
        p = os.path.join(args.qa_base_dir, args.dx, 'path2', 'stage1', 'basic')
        ensure_dir(p)
        write_json(os.path.join(p, f"{dicom}.json"), make_path2_stage1_json(dicom))

        # path2/stage2/basic
        p = os.path.join(args.qa_base_dir, args.dx, 'path2', 'stage2', 'basic')
        ensure_dir(p)
        write_json(os.path.join(p, f"{dicom}.json"), make_path2_stage2_json(dicom))

        # path2/stage3/basic
        p = os.path.join(args.qa_base_dir, args.dx, 'path2', 'stage3', 'basic')
        ensure_dir(p)
        write_json(os.path.join(p, f"{dicom}.json"), make_path2_stage3_json(dicom))

        # re-path1/init/basic
        p = os.path.join(args.qa_base_dir, args.dx, 're-path1', 'init', 'basic')
        ensure_dir(p)
        write_json(os.path.join(p, f"{dicom}.json"), make_repath1_init_json(dicom))

        # re-path1/stage1/basic
        p = os.path.join(args.qa_base_dir, args.dx, 're-path1', 'stage1', 'basic')
        ensure_dir(p)
        write_json(os.path.join(p, f"{dicom}.json"), make_repath1_stage1_json(dicom))

        # re-path1/stage2/basic
        p = os.path.join(args.qa_base_dir, args.dx, 're-path1', 'stage2', 'basic')
        ensure_dir(p)
        write_json(os.path.join(p, f"{dicom}.json"), make_repath1_stage2_json(dicom))

        # re-path1/stage3/basic
        p = os.path.join(args.qa_base_dir, args.dx, 're-path1', 'stage3', 'basic')
        ensure_dir(p)
        write_json(os.path.join(p, f"{dicom}.json"), make_repath1_stage3_json(dicom))

        # re-path1/stage4/basic
        p = os.path.join(args.qa_base_dir, args.dx, 're-path1', 'stage4', 'basic')
        ensure_dir(p)
        write_json(os.path.join(p, f"{dicom}.json"), make_repath1_stage4_json(dicom))

    # write out dx_by_dicoms for the sample
    out_map = {args.dx: sampled}
    with open(args.out_dx_by_dicoms, 'w', encoding='utf-8') as fh:
        json.dump(out_map, fh, indent=2)

    print(f'Wrote minimal QA for {len(sampled)} dicoms under {args.qa_base_dir} and saved list to {args.out_dx_by_dicoms}')

if __name__ == '__main__':
    main()
