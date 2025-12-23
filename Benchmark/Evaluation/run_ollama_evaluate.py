import os
import subprocess
import argparse
import sys


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxreasonbench_base_dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output_nih')), type=str)
    parser.add_argument('--model', required=True, help='Ollama model name to evaluate (e.g. my-model)')
    parser.add_argument('--scorer_model', required=True, help='Ollama model name to use for scoring')
    parser.add_argument('--ollama_use_gpu', action='store_true', help='Request Ollama CLI to use GPU (if supported)')
    parser.add_argument('--test_limit', default=None, type=int, help='If set, limit number of dicoms per dx for quick test')
    parser.add_argument('--save_base_dir', default='result_ollama', type=str)
    parser.add_argument('--tensor_parallel_size', default='1', type=str)
    parser.add_argument('--nih_cxr_base', default=None, type=str, help='Optional NIH dataset root (will be passed to evaluate_reasoning.py)')
    parser.add_argument('--dx_by_dicoms_file', default=None, type=str, help='Optional dx_by_dicoms.json to pass to evaluate_reasoning.py')
    parser.add_argument('--force_all_guidance', action='store_true', help='Forward to evaluate_guidance.py to force running guidance for all dicoms')
    parser.add_argument('--no_review', action='store_true', help='Forward to evaluate_guidance.py to skip review steps')
    return parser.parse_args()


def run_evaluation():
    args = config()
    eval_dir = os.path.dirname(os.path.abspath(__file__))

    model_id = f"ollama:{args.model}"
    model_id4scoring = f"ollama:{args.scorer_model}"
    # sanitized short names for filesystem paths (strip 'ollama:' prefix)
    model_id_short_for_paths = model_id.split(':', 1)[1] if ':' in model_id else model_id
    model_id4scoring_short_for_paths = model_id4scoring.split(':', 1)[1] if ':' in model_id4scoring else model_id4scoring

    # Check if ollama CLI exists and whether models are available; try to pull if missing
    def ensure_ollama_model(model_name):
        import shutil
        import sys
        if shutil.which('ollama') is None:
            print('Warning: ollama CLI not found in PATH. Ensure ollama is installed and daemon is running.')
            return

        # model_name may be 'ollama:my-model' or 'my-model'
        model_short = model_name.split(':', 1)[1] if ':' in model_name else model_name
        try:
            # List models and see if model exists
            res = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
            if model_short in res.stdout:
                print(f"Ollama model '{model_short}' already present.")
                return
            print(f"Ollama model '{model_short}' not found locally — attempting to pull...")
            pull = subprocess.run(["ollama", "pull", model_short], capture_output=True, text=True)
            if pull.returncode == 0:
                print(f"Successfully pulled Ollama model '{model_short}'.")
            else:
                print(f"Failed to pull Ollama model '{model_short}': {pull.stderr}")
                print("Exiting due to failed model pull.")
                sys.exit(1)
        except Exception as e:
            print(f"Error checking/pulling Ollama model '{model_short}': {e}")

    ensure_ollama_model(model_id)
    ensure_ollama_model(model_id4scoring)

    # Try to trigger model preload on Ollama daemon by issuing a short-run with timeout.
    # This helps ensure the daemon loads the model (and uses GPU if configured) before long-running evaluation.
    def try_preload_ollama_model(model_name, timeout_sec=20):
        import subprocess
        model_short = model_name.split(':', 1)[1] if ':' in model_name else model_name
        try:
            proc = subprocess.run(["ollama", "run", model_short], input="\n", capture_output=True, text=True, timeout=timeout_sec)
            if proc.returncode == 0:
                print(f"Preload check: model '{model_short}' responded quickly.")
            else:
                print(f"Preload check: model '{model_short}' returned code {proc.returncode} (stderr: {proc.stderr.strip()}).")
        except subprocess.TimeoutExpired:
            print(f"Preload timeout: model '{model_short}' did not finish within {timeout_sec}s; it may be loading in daemon.")
        except FileNotFoundError:
            print("Warning: 'ollama' command not found for preload check.")
        except Exception as e:
            print(f"Preload check error for model '{model_short}': {e}")

    # Attempt to preload both models (non-fatal)
    try_preload_ollama_model(model_id)
    try_preload_ollama_model(model_id4scoring)

    reasoning_args = [
        sys.executable,
        os.path.join(eval_dir, "evaluate_reasoning.py"),
        "--model_id", model_id,
        "--model_path", "",
        "--tensor_parallel_size", args.tensor_parallel_size,
        "--model_id4scoring", model_id4scoring,
        "--cxreasonbench_base_dir", args.cxreasonbench_base_dir,
    ]
    # prefer explicit NIH dataset root if provided
    if args.nih_cxr_base:
        reasoning_args += ["--nih_cxr_base", args.nih_cxr_base]
    else:
        reasoning_args += ["--mimic_cxr_base", args.cxreasonbench_base_dir]

    # always pass save_base_dir
    reasoning_args += ["--save_base_dir", args.save_base_dir]
    # optionally pass a custom dx_by_dicoms_file to focus the run
    if getattr(args, 'dx_by_dicoms_file', None):
        reasoning_args += ["--dx_by_dicoms_file", args.dx_by_dicoms_file]
    # ensure GOOGLE_CLOUD_* are passed as strings (avoid None causing TypeError in set_gcp_env)
    reasoning_args += [
        "--GOOGLE_CLOUD_LOCATION", "",
        "--GOOGLE_CLOUD_PROJECT", "",
        "--GOOGLE_GENAI_USE_VERTEXAI", "False",
        "--TOKENIZERS_PARALLELISM", "false",
    ]
    if args.ollama_use_gpu:
        reasoning_args += ["--ollama_use_gpu"]
    if args.test_limit is not None:
        reasoning_args += ["--test_limit", str(args.test_limit)]

    print("Running evaluate_reasoning.py ...")
    subprocess.run(reasoning_args, check=True)

    save_base_dir = args.save_base_dir
    model_id_short = model_id_short_for_paths
    model_id4scoring_short = model_id4scoring_short_for_paths
    config_path = os.path.join(save_base_dir, 'inference', 'reasoning', model_id_short, "config.json")

    guidance_args = [
        sys.executable,
        os.path.join(eval_dir, "evaluate_guidance.py"),
        "--config_path", config_path,
    ]
    # forward guidance control flags if requested
    if getattr(args, 'force_all_guidance', False):
        guidance_args += ["--force_all_guidance"]
    if getattr(args, 'no_review', False):
        guidance_args += ["--no_review"]

    print("Running evaluate_guidance.py ...")
    subprocess.run(guidance_args, check=True)

    for evaluation_path in ['reasoning', 'guidance']:
        saved_dir_inference = os.path.join(save_base_dir, 'inference', evaluation_path, model_id_short)
        saved_dir_scoring = os.path.join(save_base_dir, 'scoring', evaluation_path, model_id4scoring_short, model_id_short)
        metric_args = [
            sys.executable,
            os.path.join(eval_dir, "metric.py"),
            "--saved_dir_inference", saved_dir_inference,
            "--saved_dir_scoring", saved_dir_scoring,
        ]
        print(f"Running metric.py for {evaluation_path} ...")
        subprocess.run(metric_args, check=True)


if __name__ == "__main__":
    run_evaluation()
