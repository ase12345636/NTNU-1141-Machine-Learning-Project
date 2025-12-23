import os
import random
import argparse
import json
import pandas as pd
from types import SimpleNamespace

from generate_benchmark import (
    qa_init_question, qa_criteria, qa_custom_criteria, qa_bodypart,
    qa_measurement, qa_final, set_dataset_config, DatasetConfig, initial_question_per_dx
)


def make_measured_value_for(target):
    if target == 'projection':
        return [0.2, 0.2]
    if target == 'inclusion':
        return ['_in']
    if target == 'inspiration':
        return 10
    if target == 'trachea_deviation':
        return 'deviated'
    if target in ['rotation', 'aortic_knob_enlargement', 'ascending_aorta_enlargement',
                  'descending_aorta_enlargement', 'descending_aorta_tortuous',
                  'cardiomegaly', 'mediastinal_widening', 'carina_angle']:
        return 0.5
    return 0


def main():
    parser = argparse.ArgumentParser(description='Run QA smoke test')
    parser.add_argument('--data_csv', default=None, help='Path to Data_Entry_2017.csv')
    parser.add_argument('--chexstruct_base_dir', default=None, help='Path to directory containing per-dx CSV files (e.g., cardiomegaly.csv)')
    args_cli = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    # resolve data CSV path
    if args_cli.data_csv:
        csv_path = os.path.abspath(args_cli.data_csv)
    else:
        csv_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'dataset', 'Data_Entry_2017.csv'))

    if not os.path.exists(csv_path):
        print(f"Data CSV not found at {csv_path}. Aborting smoke test.")
        return

    df = pd.read_csv(csv_path)
    img_col = None
    for c in df.columns:
        lc = c.lower()
        if 'image' in lc or 'filename' in lc or 'file' in lc or 'index' in lc:
            img_col = c
            break
    if img_col is None:
        img_col = df.columns[0]

    images = df[img_col].astype(str).tolist()
    dicoms = [os.path.splitext(os.path.basename(x.strip()))[0] for x in images if x.strip()]
    dicoms = list(dict.fromkeys(dicoms))

    n = min(50, len(dicoms))
    if n == 0:
        print("No dicom identifiers found in CSV. Aborting.")
        return

    sample = random.sample(dicoms, n)

    # configure dataset as NIH
    chexstruct_dir = None
    # prefer explicit CLI arg
    if args_cli.chexstruct_base_dir:
        chexstruct_dir = os.path.abspath(args_cli.chexstruct_base_dir)
    else:
        # try common locations relative to repo
        # try common locations relative to repo (including nih_cxr14 which contains per-dx CSVs)
        cand = [
            os.path.abspath(os.path.join(base_dir, '..', '..', 'CheXStruct', 'diagnostic_tasks')),
            os.path.abspath(os.path.join(base_dir, '..', '..', 'chexstruct', 'diagnostic_tasks')),
            os.path.abspath(os.path.join(base_dir, '..', '..', 'cxreasonbench_chexstruct', 'diagnostic_tasks')),
            os.path.abspath(os.path.join(base_dir, '..', '..', 'nih_cxr14')),
            os.path.abspath(os.path.join(base_dir, '..', '..', 'nih-cxr14_viz')),
            os.path.abspath(os.path.join(base_dir, '..', '..', 'CheXStruct')),
            os.path.abspath(os.path.join(base_dir, '..', '..'))
        ]

        for c in cand:
            if not os.path.isdir(c):
                continue
            # quick heuristic: check for any dx csv (recursive)
            has_csv = False
            for root, dirs, files in os.walk(c):
                for fname in files:
                    if fname.lower().endswith('.csv'):
                        has_csv = True
                        break
                if has_csv:
                    break
            if has_csv:
                chexstruct_dir = c
                break

    if chexstruct_dir is None:
        print('Warning: chexstruct_base_dir not found. You can pass --chexstruct_base_dir to point to the directory containing <dx>.csv files.')
        chexstruct_dir = ''

    args = SimpleNamespace(dataset_name='nih-cxr14', chexstruct_base_dir=chexstruct_dir)
    set_dataset_config(DatasetConfig(args))

    out_dir = os.path.join(base_dir, 'qa_smoke_test_output')
    os.makedirs(out_dir, exist_ok=True)

    success = 0
    failures = []
    missing_images = []
    per_sample_status = {}

    # locate dataset image root for existence checks
    dataset_root = os.path.abspath(os.path.join(base_dir, '..', '..', 'dataset'))

    def image_exists(dicom_id):
        # search for common image filename variants under dataset_root
        if not os.path.isdir(dataset_root):
            return False
        for root, dirs, files in os.walk(dataset_root):
            fname_png = f"{dicom_id}.png"
            fname_jpg = f"{dicom_id}.jpg"
            if fname_png in files or fname_jpg in files:
                return True
        return False

    # QA generation parameters
    num_options = 5
    pitfall = 'basic'
    inference_type = 'reasoning'
    segmask_base_dir = os.path.abspath(os.path.join(base_dir, '..', '..', 'output_nih', 'segmask_bodypart'))

    for dicom in sample:
        target = random.choice(list(initial_question_per_dx.keys()))
        measured = make_measured_value_for(target)

        # skip if image file missing to avoid generating phantom QA
        if not image_exists(dicom):
            missing_images.append(dicom)
            failures.append((dicom, target, 'image_missing'))
            continue

        stage_order = ['init', 'criteria', 'custom_criteria', 'bodypart', 'measurement', 'final']
        stage_results = {s: None for s in stage_order}

        # helper to check whether a json for this dicom was produced under out_dir
        def produced_json_exists(dicom_id):
            for root, dirs, files in os.walk(out_dir):
                if f"{dicom_id}.json" in files:
                    return True, os.path.join(root, f"{dicom_id}.json")
            return False, None

        try:
            # init
            try:
                answer_init = qa_init_question('smoke', dicom, target, measured, mimic_meta=None, save_dir_gold_qa=out_dir)
                if answer_init is None:
                    stage_results['init'] = ('skipped', 'image_missing')
                    # If init skipped (e.g., missing image), do not run later stages
                    per_sample_status[dicom] = {'target': target, 'measured': measured, 'stages': stage_results}
                    missing_images.append(dicom)
                    failures.append((dicom, target, 'image_missing'))
                    continue
                else:
                    ok, path = produced_json_exists(dicom)
                    stage_results['init'] = ('ok', path if ok else 'no_json')
            except Exception as e:
                stage_results['init'] = ('error', str(e))

            # criteria
            try:
                qa_criteria('smoke', dicom, target, num_options, pitfall, save_dir_gold_qa=out_dir)
                ok, path = produced_json_exists(dicom)
                stage_results['criteria'] = ('ok', path if ok else 'no_json')
            except Exception as e:
                stage_results['criteria'] = ('error', str(e))

            # custom/refined criteria
            try:
                qa_custom_criteria('smoke', dicom, target, save_dir_gold_qa=out_dir)
                ok, path = produced_json_exists(dicom)
                stage_results['custom_criteria'] = ('ok', path if ok else 'no_json')
            except Exception as e:
                stage_results['custom_criteria'] = ('error', str(e))

            # bodypart selection
            try:
                qa_bodypart('smoke', inference_type, dicom, target, num_options, pitfall, segmask_base_dir, save_dir_gold_qa=out_dir)
                ok, path = produced_json_exists(dicom)
                stage_results['bodypart'] = ('ok', path if ok else 'no_json')
            except Exception as e:
                stage_results['bodypart'] = ('error', str(e))

            # measurement (if applicable)
            try:
                qa_measurement('smoke', inference_type, dicom, target, measured, num_options, save_dir_gold_qa=out_dir)
                ok, path = produced_json_exists(dicom)
                stage_results['measurement'] = ('ok', path if ok else 'no_json')
            except Exception as e:
                stage_results['measurement'] = ('error', str(e))

            # final
            try:
                qa_final('smoke', dicom, target, measured, answer_init, save_dir_gold_qa=out_dir)
                ok, path = produced_json_exists(dicom)
                stage_results['final'] = ('ok', path if ok else 'no_json')
            except Exception as e:
                stage_results['final'] = ('error', str(e))

            # determine overall result
            # Enforce per-stage consistency: expected files for each stage
            expected_files = [
                os.path.join(out_dir, 'init', 'basic', f"{dicom}.json"),
                os.path.join(out_dir, 'criteria', pitfall, f"{dicom}.json"),
                os.path.join(out_dir, 'custom_criteria', 'basic', f"{dicom}.json"),
                os.path.join(out_dir, 'bodypart', pitfall, f"{dicom}.json"),
                os.path.join(out_dir, 'measurement', 'basic', f"{dicom}.json"),
                os.path.join(out_dir, 'final', 'basic', f"{dicom}.json"),
            ]
            exists = [os.path.exists(p) for p in expected_files]

            # If partially present (some True, some False), remove existing partial files and mark as failure
            if any(exists) and not all(exists):
                # remove partial outputs for this dicom
                for p, e in zip(expected_files, exists):
                    if e:
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                # mark as failure due to incomplete generation
                stage_results = {s: ('skipped', 'incomplete') for s in stage_order}
                failures.append((dicom, target, stage_results))
            else:
                if any(v[0] == 'error' for v in stage_results.values() if v):
                    failures.append((dicom, target, stage_results))
                else:
                    success += 1

        except Exception as e:
            failures.append((dicom, target, str(e)))

        per_sample_status[dicom] = {'target': target, 'measured': measured, 'stages': stage_results}

    print(f"Smoke test finished. Generated {success} QA items. {len(failures)} failures.")
    if failures:
        print("Failures (dicom, target, error):")
        for f in failures:
            print(f)
    # write summary JSON
    summary = {
        'generated': success,
        'failures': len(failures),
        'missing_images': missing_images,
        'failures_detail': failures,
        'per_sample_status': per_sample_status,
    }
    with open(os.path.join(out_dir, 'smoke_summary.json'), 'w') as sf:
        json.dump(summary, sf, indent=2)

    print(f"QA JSON files (per-stage) written under: {out_dir}")
    print(f"Summary written to: {os.path.join(out_dir, 'smoke_summary.json')}")


if __name__ == '__main__':
    main()
