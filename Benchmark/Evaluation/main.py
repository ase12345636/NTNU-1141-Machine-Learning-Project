import os
import subprocess
import argparse
import json


def run_evaluation():
    eval_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str,
                        help='Optional path to a JSON config to populate evaluate_reasoning args')
    args, _ = parser.parse_known_args()

    # Build default reasoning args
    reasoning_args = [
        "python",
        os.path.join(eval_dir, "evaluate_reasoning.py"),
        "--model_id", "",
        "--model_path", "",
        "--tensor_parallel_size", "2",
        "--model_id4scoring", "gemini-2.0-flash",
        "--cxreasonbench_base_dir", "",
        "--mimic_cxr_base", "",
        "--save_base_dir", "result"
    ]

    # If a config file is provided, load known keys and replace defaults
    if args.config:
        try:
            with open(args.config, 'r') as fh:
                cfg = json.load(fh)
            def set_arg(opt, val):
                try:
                    idx = reasoning_args.index(opt)
                    reasoning_args[idx + 1] = str(val) if val is not None else ""
                except ValueError:
                    pass

            set_arg("--model_id", cfg.get('model_id', reasoning_args[reasoning_args.index('--model_id')+1]))
            set_arg("--model_path", cfg.get('model_path', reasoning_args[reasoning_args.index('--model_path')+1]))
            set_arg("--tensor_parallel_size", cfg.get('tensor_parallel_size', reasoning_args[reasoning_args.index('--tensor_parallel_size')+1]))
            set_arg("--model_id4scoring", cfg.get('model_id4scoring', reasoning_args[reasoning_args.index('--model_id4scoring')+1]))
            set_arg("--cxreasonbench_base_dir", cfg.get('cxreasonbench_base_dir', reasoning_args[reasoning_args.index('--cxreasonbench_base_dir')+1]))
            set_arg("--mimic_cxr_base", cfg.get('mimic_cxr_base', reasoning_args[reasoning_args.index('--mimic_cxr_base')+1]))
            set_arg("--save_base_dir", cfg.get('save_base_dir', reasoning_args[reasoning_args.index('--save_base_dir')+1]))
        except Exception as e:
            print(f"Warning: failed to load config {args.config}: {e}")

    print("Running evaluate_reasoning.py ...")
    subprocess.run(reasoning_args, check=True)

    # -------------------------
    #  2️⃣ evaluate_guidance.py
    # -------------------------
    save_base_dir = reasoning_args[reasoning_args.index("--save_base_dir") + 1]
    model_id = reasoning_args[reasoning_args.index("--model_id") + 1]
    config_path = os.path.join(save_base_dir, 'inference/reasoning', model_id, "config.json")

    guidance_args = [
        "python",
        os.path.join(eval_dir, "evaluate_guidance.py"),
        "--config_path", config_path,
    ]
    print("Running evaluate_guidance.py ...")
    subprocess.run(guidance_args, check=True)

    # -------------------------
    #      3️⃣ metric.py
    # -------------------------
    model_id4scoring = reasoning_args[reasoning_args.index("--model_id4scoring") + 1]
    for evaluation_path in ['reasoning', 'guidance']:
        saved_dir_inference = os.path.join(save_base_dir, 'inference', evaluation_path, model_id)
        saved_dir_scoring = os.path.join(save_base_dir, 'scoring', evaluation_path, model_id4scoring, model_id)
        metric_args = [
            "python",
            os.path.join(eval_dir, "metric.py"),
            "--saved_dir_inference", saved_dir_inference,
            "--saved_dir_scoring", saved_dir_scoring,
        ]
        print("Running metric.py ...")
        subprocess.run(metric_args, check=True)

if __name__ == "__main__":
    run_evaluation()