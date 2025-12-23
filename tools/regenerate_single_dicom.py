"""
Regenerate QA for a single dicom using Benchmark/generation/generate_benchmark.py
Usage:
    python tools/regenerate_single_dicom.py \
      --dx aortic_knob_enlargement --dicom 00000001_000 --inference_path path1 \
      --dataset_name nih-cxr14 --save_base d:/CXReasonBench/output_nih --cxreasonbench_base_dir d:/CXReasonBench \
      --chexstruct_base_dir d:/CXReasonBench/nih_cxr14

This imports the generator module and calls the worker for one task.
"""
import os
import argparse
import importlib.util


def load_module_from_path(path):
    spec = importlib.util.spec_from_file_location('generate_benchmark', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_args_namespace(args):
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
    a.workers = 1
    a.segmask_base_dir = args.segmask_base_dir or os.path.join(a.cxreasonbench_base_dir, 'output_nih', 'segmask_bodypart')
    a.pnt_base_dir = os.path.join(a.cxreasonbench_base_dir, 'output_nih', 'pnt_on_cxr')
    return a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dx', required=True)
    parser.add_argument('--dicom', required=True)
    parser.add_argument('--generate_benchmark_py', default=os.path.join(os.path.dirname(__file__), '..', 'Benchmark', 'generation', 'generate_benchmark.py'))
    parser.add_argument('--inference_path', default='path1')
    parser.add_argument('--dataset_name', default='nih-cxr14')
    parser.add_argument('--save_base', default=os.path.join(os.path.dirname(__file__), '..', 'output_nih'))
    parser.add_argument('--chexstruct_base_dir', default=os.path.join(os.path.dirname(__file__), '..', 'nih_cxr14'))
    parser.add_argument('--cxreasonbench_base_dir', default=os.path.join(os.path.dirname(__file__), '..'))
    parser.add_argument('--nih_image_base_dir', default=os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    parser.add_argument('--num_options', default=5, type=int)
    parser.add_argument('--segmask_base_dir', default=None)
    args = parser.parse_args()

    gen_path = os.path.abspath(args.generate_benchmark_py)
    if not os.path.exists(gen_path):
        print('generate_benchmark.py not found at', gen_path)
        return

    mod = load_module_from_path(gen_path)
    mod_args = build_args_namespace(args)
    mimic_meta = None
    dataset_config = mod.DatasetConfig(mod_args, mimic_meta)
    mod.set_dataset_config(dataset_config)

    task = (args.dx, args.dicom, 0)
    print('Regenerating', task)
    result = mod._process_single_dicom(task)
    print('Result:', result)


if __name__ == '__main__':
    main()
