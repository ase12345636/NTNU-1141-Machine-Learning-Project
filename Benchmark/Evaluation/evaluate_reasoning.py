import os
import json
import argparse
import sys
from tqdm import tqdm
from glob import glob
from datetime import datetime

from utils import set_seed, set_gcp_env, dx_task_measurement, dx_task_multi_bodyparts
from prompt import system_message, stage_by_sysmsg
from scoring import return_scoring_result, return_measured_value_result

from model_cards import load_model_n_prosessor, inference_vllms


def config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--shot', default=None, type=int)
    parser.add_argument('--img_size', default=1024, type=int)
    parser.add_argument('--tensor_parallel_size', default=2, type=int)
    parser.add_argument('--evaluation_path', default='reasoning', type=str)
    parser.add_argument('--model_id4scoring', default="gemini-2.0-flash", type=str)
    parser.add_argument('--test_limit', default=None, type=int, help='Limit number of dicoms per diagnostic task for testing')
    parser.add_argument('--ollama_use_gpu', action='store_true', help='Request Ollama CLI to use GPU (if supported)')
    parser.add_argument('--dx_by_dicoms_file', default=None, type=str, help='Optional path to a dx_by_dicoms.json to override the default')


    parser.add_argument('--model_id', default="Qwen/Qwen2.5-VL-7B-Instruct", type=str,
                        help='Model identifier to evaluate (supports arbitrary strings including ollama:<name>)')
    parser.add_argument('--model_path', default='path/to/saved_model_path', type=str)

    parser.add_argument('--cxreasonbench_base_dir', default="", type=str)
    parser.add_argument('--mimic_cxr_base', default="", type=str)
    parser.add_argument('--nih_cxr_base', default=None, type=str,
                        help='Alias for --mimic_cxr_base to point to NIH dataset root')

    parser.add_argument('--save_base_dir', default='result', type=str)

    parser.add_argument('--GOOGLE_CLOUD_LOCATION', default=None, type=str)
    parser.add_argument('--GOOGLE_CLOUD_PROJECT', default=None, type=str)
    parser.add_argument('--GOOGLE_GENAI_USE_VERTEXAI', default="True", type=str)
    parser.add_argument('--TOKENIZERS_PARALLELISM', default='false', type=str)

    parser.add_argument('--gpt_endpoint', default=None, type=str)
    parser.add_argument('--gpt_api_key', default=None, type=str)
    parser.add_argument('--gpt_api_version', default="2025-01-01-preview", type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = config()
    args_dict = vars(args)

# Backward-compatible handling: allow --nih_cxr_base as alias for --mimic_cxr_base
    if getattr(args, 'nih_cxr_base', None):
        # prefer explicit --mimic_cxr_base if both provided, otherwise copy
        if not args.mimic_cxr_base:
            args.mimic_cxr_base = args.nih_cxr_base
    else:
        # keep existing behavior if nih_cxr_base not provided
        pass

    args.segmask_base_dir = os.path.join(args.cxreasonbench_base_dir, 'segmask_bodypart')
    args.pnt_base_dir = os.path.join(args.cxreasonbench_base_dir, 'pnt_on_cxr')
    args.qa_base_dir = os.path.join(args.cxreasonbench_base_dir, 'qa')
    # allow overriding dx_by_dicoms.json via CLI for focused testing
    if args.dx_by_dicoms_file is None:
        args.dx_by_dicoms_file = os.path.join(args.cxreasonbench_base_dir, 'dx_by_dicoms.json')

    set_gcp_env(args)
    set_seed(args.seed)

    # sanitize model ids for filesystem paths (avoid characters like ':' on mounted drives)
    safe_model_id = args.model_id.split(':', 1)[1] if ':' in args.model_id else args.model_id
    safe_model_id4scoring = args.model_id4scoring.split(':', 1)[1] if ':' in args.model_id4scoring else args.model_id4scoring

    save_dir_reasoning = os.path.join(args.save_base_dir, 'inference', args.evaluation_path, safe_model_id)
    save_dir_reasoning_scoring = os.path.join(args.save_base_dir, 'scoring', args.evaluation_path, safe_model_id4scoring, safe_model_id)

    args.save_dir_reasoning = save_dir_reasoning
    args.save_dir_reasoning_scoring = save_dir_reasoning_scoring

    os.makedirs(save_dir_reasoning, exist_ok=True)
    os.makedirs(save_dir_reasoning_scoring, exist_ok=True)

    diagnostic_task_list = ['aortic_knob_enlargement', 'ascending_aorta_enlargement',
                            'cardiomegaly', 'carina_angle', 'descending_aorta_enlargement',
                            'descending_aorta_tortuous', 'inclusion', 'inspiration',
                            'mediastinal_widening', 'projection', 'rotation', 'trachea_deviation']
    model, processor = load_model_n_prosessor(args, args.model_id, args.model_path)

    # Precompute total remaining dicoms for an overall progress bar
    with open(args.dx_by_dicoms_file, "r") as file:
        dx_by_dicoms_all = json.load(file)
    total_remained = 0
    remained_map = {}
    for diagnostic_task in diagnostic_task_list:
        dicom_lst = dx_by_dicoms_all.get(diagnostic_task, [])
        save_dir_per_dx = os.path.join(save_dir_reasoning, diagnostic_task)
        saved_dicom_lst = [path.split('/')[-1].split('.')[0] for path in glob(f"{save_dir_per_dx}/*.json")]
        remained_dicom_lst = list(set(dicom_lst).difference(saved_dicom_lst))
        remained_map[diagnostic_task] = remained_dicom_lst
        total_remained += len(remained_dicom_lst)

    # Build a global filename -> fullpath index once to avoid repeated expensive os.walk
    # Prefer explicit mimic/nih dataset roots, otherwise fall back to the provided
    # `cxreasonbench_base_dir` which contains `pnt_on_cxr`, `qa`, `segmask_bodypart`, etc.
    dataset_root_for_index = args.mimic_cxr_base or getattr(args, 'nih_cxr_base', None) or args.cxreasonbench_base_dir
    image_index = {}
    if dataset_root_for_index and os.path.exists(dataset_root_for_index):
        for dirpath, dirnames, filenames in os.walk(dataset_root_for_index):
            for fn in filenames:
                if fn not in image_index:
                    image_index[fn] = os.path.join(dirpath, fn)

    # Diagnostic: print candidate roots and first few files to help debug path issues
    candidate_roots = [args.mimic_cxr_base, getattr(args, 'nih_cxr_base', None), args.cxreasonbench_base_dir]
    candidate_roots = [r for r in candidate_roots if r]
    for r in candidate_roots:
        try:
            exists = os.path.exists(r)
            sample = None
            if exists:
                for p, d, fns in os.walk(r):
                    if fns:
                        sample = fns[:3]
                        break
            print(f"[PATH DIAG] root={r} exists={exists} sample_files={sample}")
        except Exception:
            print(f"[PATH DIAG] root={r} exists=ERROR")

    def resolve_img_paths(paths, candidate_roots):
        resolved = []
        for path in paths:
            if not path:
                continue
            if os.path.isabs(path) and os.path.exists(path):
                resolved.append(path)
                continue
            # try index by basename
            b = os.path.basename(path)
            if image_index and b in image_index:
                resolved.append(image_index[b])
                continue
            # try joining with candidate roots
            found = None
            for root in candidate_roots:
                cand = os.path.join(root, path.lstrip(os.sep))
                if os.path.exists(cand):
                    found = cand
                    break
                # try basename under root
                cand2 = os.path.join(root, b)
                if os.path.exists(cand2):
                    found = cand2
                    break
            if found:
                resolved.append(found)
            else:
                resolved.append(path)
        return resolved

    def find_image_under_root(root, fname):
        # Fast lookup via prebuilt index first
        if image_index and fname in image_index:
            return image_index[fname]
        # fast check: direct join
        cand = os.path.join(root, fname)
        if os.path.exists(cand):
            return cand
        # fallback: walk (rare, only if index not available or missing entry)
        for dirpath, dirnames, filenames in os.walk(root):
            if fname in filenames:
                return os.path.join(dirpath, fname)
        return None

    overall_bar = None
    try:
        from tqdm import tqdm as _tqdm
        overall_bar = _tqdm(total=total_remained, desc=f'Overall/{args.model_id}')
    except Exception:
        overall_bar = None

# Helper to emit warnings without breaking tqdm progress bar
def warn(msg):
    try:
        if overall_bar is not None:
            overall_bar.write(msg)
        else:
            print(msg)
    except Exception:
        print(msg)
for diagnostic_task in diagnostic_task_list:
    save_dir_per_dx = os.path.join(save_dir_reasoning, diagnostic_task)
    save_dir_per_dx_scoring = os.path.join(save_dir_reasoning_scoring, diagnostic_task)
    os.makedirs(save_dir_per_dx, exist_ok=True)
    os.makedirs(save_dir_per_dx_scoring, exist_ok=True)

    with open(f"{save_dir_reasoning}/config.json", "w") as file:
        json.dump(args_dict, file, indent=4)

    # Build a one-time image filename -> fullpath index to avoid expensive os.walk per dicom
    dataset_root_for_index = args.mimic_cxr_base or getattr(args, 'nih_cxr_base', None) or args.cxreasonbench_base_dir
    image_index = {}
    if dataset_root_for_index and os.path.exists(dataset_root_for_index):
        # Walk once per dx (can be adjusted to walk global root outside loop if desired)
        for dirpath, dirnames, filenames in os.walk(dataset_root_for_index):
            for fn in filenames:
                if fn not in image_index:
                    image_index[fn] = os.path.join(dirpath, fn)

    remained_dicom_lst = remained_map.get(diagnostic_task, [])
    if args.test_limit is not None:
        remained_dicom_lst = list(remained_dicom_lst)[:args.test_limit]

    for dicom in tqdm(remained_dicom_lst, total=len(remained_dicom_lst), desc=f'{diagnostic_task}/{args.model_id}'):
        chat_history = []
        if 'HealthGPT' in args.model_id:
            from llava import conversation as conversation_lib
            args.conv = conversation_lib.conv_templates["phi4_instruct"].copy()
        # ===========================================================================
        #                       Init Question
        # ===========================================================================
        qa_init_file = f'{args.qa_base_dir}/{diagnostic_task}/path1/init/basic/{dicom}.json'
        if not os.path.exists(qa_init_file):
            warn(f"Warning: QA init file not found: {qa_init_file}. Skipping dicom {dicom}")
            if overall_bar is not None:
                overall_bar.update(1)
            continue
        with open(qa_init_file, "r") as file:
            qa_init = json.load(file)
        answer_per_stage = {'init': qa_init['answer']}
        measured_value = ''  # qa_init['measured_value']

        # Ensure img_path is a list even if stored as a single string in JSON
        img_paths = qa_init.get('img_path', [])
        if isinstance(img_paths, str):
            img_paths = [img_paths]
        qa_init['img_path'] = img_paths
        # candidate roots for resolving images
        candidate_roots = [args.mimic_cxr_base, getattr(args, 'nih_cxr_base', None), args.cxreasonbench_base_dir]
        candidate_roots = [r for r in candidate_roots if r]
        qa_init_img_path = resolve_img_paths(qa_init['img_path'], candidate_roots)
        missing_img_paths = [p for p in qa_init_img_path if not (p and os.path.exists(p))]

        if missing_img_paths:
            warn(f"Warning: missing image files for dicom {dicom}: {missing_img_paths}. Skipping {dicom}.")
            if overall_bar is not None:
                overall_bar.update(1)
            continue
        response_init, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                            query=qa_init['question'],
                                                                            img_path_lst=qa_init_img_path,
                                                                            system_message=system_message,
                                                                            chat_history=chat_history)

        # If model indicates it cannot process images, stop early but continue to
        # write placeholders so later stages (guidance/metric) can run safely.
        if isinstance(response_init, str) and response_init.startswith('[OLLAMA_NON_VISION]'):
            warn(f"Model appears to not support image inputs: {args.model_id}. See result_ollama/debug for prompt/response files.")
            nonvision_detected = True
            nonvision_reason = response_init
            # update progress and break out of dicom loop to stop further processing
            if overall_bar is not None:
                overall_bar.update(1)
            break

        score_init, res_idx_init = return_scoring_result(args.model_id4scoring, stage_by_sysmsg['init'],
                                                         [qa_init['question']], qa_init['answer'], response_init)

        if res_idx_init is not None:
            chat_history[-1][-1] = chat_history[-1][-1][res_idx_init]

        scoring_per_stage = {'init': score_init}
        sysmsg4scoring_per_stage = {'init': stage_by_sysmsg['init']}
        if score_init != -1:
            # ===========================================================================
            #                       Criteria
            # ===========================================================================
            # Accept multiple possible folder names for stage1 (legacy names like 'criteria')
            qa_criteria_file = None
            for variant in ['stage1', 'criteria']:
                files = glob(f"{args.qa_base_dir}/{diagnostic_task}/path1/{variant}/*/{dicom}.json")
                if files:
                    qa_criteria_file = files[0]
                    break
            if qa_criteria_file is None:
                warn(f"Warning: QA criteria file not found for {dicom}. Skipping")
                if overall_bar is not None:
                    overall_bar.update(1)
                continue
            with open(qa_criteria_file, "r") as file:
                qa_criteria = json.load(file)

            score_criteria_lst = []
            for idx, (q_criteria, a_criteria) in enumerate(zip(qa_criteria['question'], qa_criteria['answer'])):
                response_criteria, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                        query=q_criteria,
                                                                                        img_path_lst=[],
                                                                                        system_message=system_message,
                                                                                        chat_history=chat_history)
                answer_per_stage[f'criteria_{idx}'] = a_criteria

                score_criteria_sub, res_idx_criteria_sub = return_scoring_result(args.model_id4scoring,
                                                                                 stage_by_sysmsg['criteria'],
                                                                                 q_criteria, a_criteria,
                                                                                 response_criteria)

                if res_idx_criteria_sub is not None:
                    chat_history[-1][-1] = chat_history[-1][-1][res_idx_criteria_sub]

                score_criteria_lst.append(score_criteria_sub)

                scoring_per_stage[f'criteria_{idx}'] = score_criteria_sub
                sysmsg4scoring_per_stage[f'criteria_{idx}'] = stage_by_sysmsg['criteria']
                if score_criteria_sub != 1:
                    break

            if len(score_criteria_lst) == sum(score_criteria_lst):
                # ===========================================================================
                #                   Custom Criteria
                # ===========================================================================
                # custom criteria can be under stage1.5 or criteria/two-round
                qa_c_criteria_file = None
                for variant in ['stage1.5', os.path.join('criteria', 'two-round'), 'stage1_5']:
                    path = f"{args.qa_base_dir}/{diagnostic_task}/path1/{variant}/basic/{dicom}.json"
                    if os.path.exists(path):
                        qa_c_criteria_file = path
                        break
                if qa_c_criteria_file is not None:
                    with open(qa_c_criteria_file, "r") as file:
                        qa_c_criteria = json.load(file)

                    response_custom_criteria, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                                   query=qa_c_criteria['question'],
                                                                                                   img_path_lst=[],
                                                                                                   system_message=system_message,
                                                                                                   chat_history=chat_history)
                    answer_per_stage[f'custom_criteria'] = qa_c_criteria['answer']

                    score_custom_criteria, res_idx_custom = return_scoring_result(args.model_id4scoring,
                                                                                  stage_by_sysmsg['custom_criteria'],
                                                                                  qa_c_criteria['question'],
                                                                                  qa_c_criteria['answer'],
                                                                                  response_custom_criteria)

                    if res_idx_custom is not None:
                        chat_history[-1][-1] = chat_history[-1][-1][res_idx_custom]

                    scoring_per_stage[f'custom_criteria'] = score_custom_criteria
                    sysmsg4scoring_per_stage[f'custom_criteria'] = stage_by_sysmsg['custom_criteria']

                else:
                    score_custom_criteria = 1

                if score_custom_criteria == 1:
                    # ===========================================================================
                    #               Body Part
                    # ===========================================================================
                    qa_bodypart_file = None
                    for variant in ['stage2', 'bodypart']:
                        files = glob(f"{args.qa_base_dir}/{diagnostic_task}/path1/{variant}/*/{dicom}.json")
                        if files:
                            qa_bodypart_file = files[0]
                            break
                    if qa_bodypart_file is None:
                        warn(f"Warning: QA bodypart file not found for {dicom}. Skipping")
                        if overall_bar is not None:
                            overall_bar.update(1)
                        continue
                    with open(qa_bodypart_file, "r") as file:
                        qa_bodypart = json.load(file)

                    score_bodypart_lst = []
                    for idx, q_bodypart in enumerate(qa_bodypart['question']):
                        img_path_lst_bodypart = [f'{args.segmask_base_dir}{path}' for path in qa_bodypart['img_path'][idx]]

                        response_bodypart, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                                query=q_bodypart,
                                                                                                img_path_lst=img_path_lst_bodypart,
                                                                                                system_message=system_message,
                                                                                                chat_history=chat_history)
                        answer_per_stage[f'bodypart_{idx}'] = qa_bodypart['answer'][idx]

                        sys_msg_bodypart = stage_by_sysmsg['bodypart_all'] if diagnostic_task in dx_task_multi_bodyparts else stage_by_sysmsg['bodypart_one']

                        score_bodypart_sub, res_idx_body = return_scoring_result(args.model_id4scoring, sys_msg_bodypart,
                                                                                 q_bodypart, qa_bodypart['answer'][idx], response_bodypart)
                        if res_idx_body is not None:
                            chat_history[-1][-1] = chat_history[-1][-1][res_idx_body]

                        score_bodypart_lst.append(score_bodypart_sub)

                        scoring_per_stage[f'bodypart_{idx}'] = score_bodypart_sub
                        sysmsg4scoring_per_stage[f'bodypart_{idx}'] = sys_msg_bodypart

                        if score_bodypart_sub != 1:
                            break

                    if len(score_bodypart_lst) == sum(score_bodypart_lst):
                        # ===========================================================================
                        #               Measurement
                        # ===========================================================================
                        qa_measurement_file = None
                        for variant in ['stage3', 'measurement']:
                            path = f"{args.qa_base_dir}/{diagnostic_task}/path1/{variant}/basic/{dicom}.json"
                            if os.path.exists(path):
                                qa_measurement_file = path
                                break
                        if qa_measurement_file is None:
                            warn(f"Warning: QA measurement file not found for {dicom}. Skipping")
                            if overall_bar is not None:
                                overall_bar.update(1)
                            continue
                        with open(qa_measurement_file, "r") as file:
                            qa_measurement = json.load(file)

                        response_measurement, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                                   query=qa_measurement['question'],
                                                                                                   img_path_lst=[],
                                                                                                   system_message=system_message,
                                                                                                   chat_history=chat_history)
                        answer_per_stage['measurement'] = qa_measurement['answer']
                        sys_msg_measurment = stage_by_sysmsg['measurement_projection'] if diagnostic_task in ['projection'] else stage_by_sysmsg['measurement']
                        score_measurement, res_idx_measure = return_scoring_result(args.model_id4scoring, sys_msg_measurment,
                                                                                   qa_measurement['question'],
                                                                                   qa_measurement['answer'],
                                                                                   response_measurement)

                        if res_idx_measure is not None:
                            chat_history[-1][-1] = chat_history[-1][-1][res_idx_measure]

                        scoring_per_stage[f'measurement'] = score_measurement
                        sysmsg4scoring_per_stage[f'measurement'] = sys_msg_measurment

                        if score_measurement == 1:
                            # ===========================================================================
                            #               Final
                            # ===========================================================================
                            qa_final_file = None
                            for variant in ['stage4', 'final']:
                                path = f"{args.qa_base_dir}/{diagnostic_task}/path1/{variant}/basic/{dicom}.json"
                                if os.path.exists(path):
                                    qa_final_file = path
                                    break
                            if qa_final_file is None:
                                warn(f"Warning: QA final file not found for {dicom}. Skipping")
                                if overall_bar is not None:
                                    overall_bar.update(1)
                                continue
                            with open(qa_final_file, "r") as file:
                                qa_final = json.load(file)
                            response_final, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                                 query=qa_final['question'],
                                                                                                 img_path_lst=[],
                                                                                                 system_message=system_message,
                                                                                                 chat_history=chat_history)
                            answer_per_stage['final'] = qa_final['answer']

                            score_final, res_idx_final = return_scoring_result(args.model_id4scoring,
                                                                               stage_by_sysmsg['final'],
                                                                               [qa_final['question']],
                                                                               qa_final['answer'],
                                                                               response_final)
                            if res_idx_final is not None:
                                chat_history[-1][-1] = chat_history[-1][-1][res_idx_final]
                                response_final = response_final[res_idx_final]

                            scoring_per_stage[f'final'] = score_final
                            sysmsg4scoring_per_stage[f'final'] = stage_by_sysmsg['final'],

                            sysmsg_value_extraction = stage_by_sysmsg['extract_value_projection'] if diagnostic_task in ['projection'] else stage_by_sysmsg['extract_value']
                            model_measured_value = return_measured_value_result(args.model_id4scoring,sysmsg_value_extraction, response_final) if diagnostic_task in dx_task_measurement else ''

                            scoring_per_stage[f'measured_value'] = model_measured_value
                            sysmsg4scoring_per_stage[f'measured_value'] = sysmsg_value_extraction

        # ====================================================================
        #                       SAVE Result
        # ====================================================================
        result = {'dicom': dicom,
                  'system_message': system_message,
                  'cxr_path': qa_init['img_path'][0],
                  'measured_value': measured_value}

        assert len(answer_per_stage) == len(chat_history)
        for stage, history in zip(answer_per_stage, chat_history):
            hist = {f'stage-{stage}': {'query': history[0],
                                       'img_path': history[1],
                                       'response': history[-1],
                                       'answer': answer_per_stage[stage]}}
            result.update(hist)

        with open(f"{save_dir_per_dx}/{dicom}.json", "w") as file:
            json.dump(result, file, indent=4)

        result_scoring = {}
        for stage, score in scoring_per_stage.items():
            result_scoring[f'stage-{stage}'] = score

        for stage, sysmsg in sysmsg4scoring_per_stage.items():
            result_scoring[f'sysmsg-{stage}'] = sysmsg

        with open(f"{save_dir_per_dx_scoring}/{dicom}.json", "w") as file:
            json.dump(result_scoring, file, indent=4)

        if scoring_per_stage['init'] == -1:
            with open(f"{save_dir_per_dx_scoring}/dicom_lst_idk_init.jsonl", "a", encoding="utf-8") as f:
                json.dump(dicom, f, ensure_ascii=False)
                f.write("\n")

        if 'custom_criteria' in scoring_per_stage:
            if scoring_per_stage['custom_criteria'] == -1:
                with open(f"{save_dir_per_dx_scoring}/dicom_lst_idk_custom.jsonl", "a", encoding="utf-8") as f:
                    json.dump(dicom, f, ensure_ascii=False)
                    f.write("\n")

        if 'final' in scoring_per_stage:
            if scoring_per_stage[f'final'] == 1:
                with open(f"{save_dir_per_dx_scoring}/dicom_lst_correct_final.jsonl", "a", encoding="utf-8") as f:
                    json.dump(dicom, f, ensure_ascii=False)
                    f.write("\n")

        # update overall progress
        if overall_bar is not None:
            overall_bar.update(1)

    # If non-vision detected, stop processing further diagnostic tasks
    if 'nonvision_detected' in globals() and nonvision_detected:
        warn(f"Stopping evaluation early due to non-vision model detection: {args.model_id}")
        break

# If we detected a non-vision reply, create placeholder reasoning outputs for remaining dicoms
if 'nonvision_detected' in globals() and nonvision_detected:
    try:
        warn("Writing placeholder reasoning outputs for remaining dicoms due to non-vision detection.")
        for diagnostic_task in diagnostic_task_list:
            save_dir_per_dx = os.path.join(save_dir_reasoning, diagnostic_task)
            os.makedirs(save_dir_per_dx, exist_ok=True)
            for dicom in remained_map.get(diagnostic_task, []):
                outfn = os.path.join(save_dir_per_dx, f"{dicom}.json")
                if os.path.exists(outfn):
                    continue
                placeholder = {
                    'dicom': dicom,
                    'system_message': system_message if 'system_message' in globals() else '',
                    'cxr_path': '',
                    'measured_value': ''
                }
                placeholder['stage-init'] = {
                    'query': '',
                    'img_path': [],
                    'response': str(nonvision_reason) if 'nonvision_reason' in globals() else '[OLLAMA_NON_VISION]',
                    'answer': ''
                }
                with open(outfn, 'w', encoding='utf-8') as f:
                    json.dump(placeholder, f, indent=4)
    except Exception:
        pass

if overall_bar is not None:
    overall_bar.close()
