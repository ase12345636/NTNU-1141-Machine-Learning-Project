"""
Driver script to generate QA using the existing `generate_benchmark.py` implementation.

Usage (example):
    python tools/run_generate_with_json.py \
        --dx_json d:/CXReasonBench/output_nih/dx_by_dicoms_real_20.json \
        --save_base d:/CXReasonBench/output_nih

This script imports `generate_benchmark.py` directly (no rule changes) and calls
its functions to generate QA files for every dicom listed in the provided JSON.
"""
import os
import sys
import json
import argparse
import importlib.util
import re


def load_module_from_path(path):
    spec = importlib.util.spec_from_file_location('generate_benchmark', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_args_namespace(args):
    # minimal Namespace-like object used by generate_benchmark
    class A: pass
    a = A()
    a.inference_path = args.inference_path
    a.dataset_name = args.dataset_name
    a.save_base_dir = args.save_base
    a.chexstruct_base_dir = args.chexstruct_base_dir
    a.cxreasonbench_base_dir = args.cxreasonbench_base_dir
    a.mimic_cxr_base = ''
    a.mimic_meta_path = ''
    a.nih_image_base_dir = args.nih_image_base_dir
    a.num_options = args.num_options
    a.workers = args.workers
    # segmask and pnt base dirs are expected by the module
    if getattr(args, 'segmask_base_dir', None):
        a.segmask_base_dir = args.segmask_base_dir
    else:
        a.segmask_base_dir = os.path.join(a.cxreasonbench_base_dir, 'segmask_bodypart')
    a.pnt_base_dir = os.path.join(a.cxreasonbench_base_dir, 'pnt_on_cxr')
    return a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dx_json', required=True, help='Path to dx_by_dicoms JSON')
    parser.add_argument('--generate_benchmark_py', default=os.path.join(os.path.dirname(__file__), '..', 'Benchmark', 'generation', 'generate_benchmark.py'), help='Path to generate_benchmark.py')
    parser.add_argument('--inference_path', default='path1', choices=['path1','path2','re-path1'])
    parser.add_argument('--dataset_name', default='nih-cxr14', choices=['nih-cxr14','mimic-cxr-jpg'])
    parser.add_argument('--save_base', default=os.path.join(os.path.dirname(__file__), '..', 'output_nih'))
    parser.add_argument('--chexstruct_base_dir', default=os.path.join(os.path.dirname(__file__), '..', 'CheXStruct'))
    parser.add_argument('--cxreasonbench_base_dir', default=os.path.join(os.path.dirname(__file__), '..'))
    parser.add_argument('--nih_image_base_dir', default=os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    parser.add_argument('--num_options', default=5, type=int)
    parser.add_argument('--workers', default=1, type=int)
    parser.add_argument('--segmask_base_dir', default=None, help='Optional explicit segmask base dir (overrides cxreasonbench_base_dir/segmask_bodypart)')
    parser.add_argument('--use_wsl', action='store_true', help='Convert Windows paths to WSL /mnt/...')
    parser.add_argument('--force', action='store_true', help='Force regenerate even if final QA exists')
    args = parser.parse_args()

    def to_wsl_path(p):
        if not p:
            return p
        # normalize separators
        p2 = p.replace('\\', '/').replace('\\\\', '/')
        m = re.match(r'^([A-Za-z]):/(.*)', p2)
        if m:
            drive = m.group(1).lower()
            rest = m.group(2).lstrip('/')
            return f'/mnt/{drive}/{rest}'
        return p2

    gen_path = os.path.abspath(args.generate_benchmark_py)
    if not os.path.exists(gen_path):
        print(f'generate_benchmark.py not found at {gen_path}')
        sys.exit(1)

    # If requested, convert user-provided Windows-style paths to WSL mount paths
    if getattr(args, 'use_wsl', False):
        args.dx_json = to_wsl_path(os.path.abspath(args.dx_json))
        args.generate_benchmark_py = to_wsl_path(os.path.abspath(args.generate_benchmark_py))
        args.save_base = to_wsl_path(os.path.abspath(args.save_base))
        args.chexstruct_base_dir = to_wsl_path(os.path.abspath(args.chexstruct_base_dir))
        args.cxreasonbench_base_dir = to_wsl_path(os.path.abspath(args.cxreasonbench_base_dir))
        args.nih_image_base_dir = to_wsl_path(os.path.abspath(args.nih_image_base_dir))
        gen_path = args.generate_benchmark_py

    mod = load_module_from_path(gen_path)

    # Build minimal args namespace expected by the module
    mod_args = build_args_namespace(args)

    # Load dx json
    with open(args.dx_json, 'r', encoding='utf-8') as f:
        dx_by_dicoms = json.load(f)

    # Prepare dataset config (mimic_meta not needed for NIH)
    mimic_meta = None
    dataset_config = mod.DatasetConfig(mod_args, mimic_meta)
    mod.set_dataset_config(dataset_config)

    # Ensure save base dir exists
    os.makedirs(mod_args.save_base_dir, exist_ok=True)

    # Process each dx/dicom sequentially using provided functions (no custom rules)
    # Populate MC format cache and call the worker function
    for dx, dicom_list in dx_by_dicoms.items():
        print(f'Processing {dx} ({len(dicom_list)} images)')
        # Diagnostic: check segmask directory and sample files
        segmask_dir = mod_args.segmask_base_dir
        dx_seg_dir = os.path.join(segmask_dir, dx)
        if not os.path.isdir(segmask_dir):
            print(f"Warning: segmask base dir does not exist: {segmask_dir}")
        elif not os.path.isdir(dx_seg_dir):
            print(f"Warning: segmask dx dir does not exist: {dx_seg_dir}")
        else:
            # list a few subdirs and a few example files
            try:
                subdirs = [d for d in os.listdir(dx_seg_dir) if os.path.isdir(os.path.join(dx_seg_dir, d))]
                print(f"  segmask subdirs count: {len(subdirs)}; example subdirs: {subdirs[:5]}")
                # find one png for example if present
                example_found = False
                for sd in subdirs[:10]:
                    sd_path = os.path.join(dx_seg_dir, sd)
                    files = [f for f in os.listdir(sd_path) if f.endswith('.png')]
                    if files:
                        print(f"  example file: {os.path.join(sd_path, files[0])}")
                        example_found = True
                        break
                if not example_found:
                    print(f"  No .png files found under {dx_seg_dir} subdirs (this may cause skip-noseg)")
            except Exception as e:
                print(f"  Could not list segmask dir {dx_seg_dir}: {e}")
        mc_fmt_crit, mc_fmt_bodypart = mod.return_mc_format_list(mod_args, dx)
        # cache for workers
        mod._MC_FORMATS[dx] = (mc_fmt_crit, mc_fmt_bodypart)

        for idx, dicom in enumerate(dicom_list):
            task = (dx, dicom, idx)
            # If forcing, remove existing final QA to force regeneration
            if getattr(args, 'force', False):
                save_dir_qa = os.path.join(mod_args.save_base_dir, 'qa', dx, mod_args.inference_path)
                final_q_path = os.path.join(save_dir_qa, 'final', 'basic', f"{dicom}.json")
                try:
                    if os.path.exists(final_q_path):
                        os.remove(final_q_path)
                        print(f"  removed existing final QA to force regenerate: {final_q_path}")
                except Exception as e:
                    print(f"  warning: could not remove {final_q_path}: {e}")

            try:
                dx_r, dicom_r, status = mod._process_single_dicom(task)
            except Exception as e:
                print(f'Error processing {dx}/{dicom}: {e}')
                status = 'error'
            if status != 'ok':
                print(f'  [{status}] {dx}/{dicom}')
            else:
                print(f'  [ok] {dx}/{dicom}')

    print('Done.')


if __name__ == '__main__':
    main()
