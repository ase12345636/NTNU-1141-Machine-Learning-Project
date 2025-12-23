import os
import json
import time
import random
import argparse
from glob import glob
from tqdm import tqdm

from utils import set_seed, set_gcp_env, dx_task_measurement, dx_task_multi_bodyparts
from prompt import stage_by_sysmsg
from scoring import return_scoring_result, return_measured_value_result

from model_cards import load_model_n_prosessor, inference_vllms


def config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', default='path/to/saved/config.json', type=str)
    parser.add_argument('--test_limit', default=None, type=int, help='Limit number of dicoms per dx for quick test')
    parser.add_argument('--force_all_guidance', action='store_true', help='Run guidance for all dicoms present in reasoning outputs instead of only idk lists')
    parser.add_argument('--no_review', action='store_true', help='Do not perform review steps after guidance (disable cross-dicom memory/review)')


    args = parser.parse_args()
    return args

def load_jsonl_if_exists(file_path):
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    return []


def warn(msg):
    try:
        print(msg)
    except Exception:
        pass

if __name__ == '__main__':
    args = config()

    with open(args.config_path, "r") as f:
        loaded_dict = json.load(f)

    vars(args).update(loaded_dict)
    args.evaluation_path = 'guidance'

    args_dict = vars(args)

    # sanitize model ids for filesystem paths (avoid characters like ':' on mounted drives)
    safe_model_id = args.model_id.split(':', 1)[1] if ':' in args.model_id else args.model_id
    safe_model_id4scoring = args.model_id4scoring.split(':', 1)[1] if ':' in args.model_id4scoring else args.model_id4scoring

    save_dir_guidance = os.path.join(args.save_base_dir, 'inference', 'guidance', safe_model_id)
    os.makedirs(save_dir_guidance, exist_ok=True)
    with open(f"{save_dir_guidance}/config.json", "w") as file:
        json.dump(args_dict, file, indent=4)

    set_gcp_env(args)
    set_seed(args.seed)

    saved_dir_reasoning = args.save_dir_reasoning
    saved_dir_scoring = f"{args.save_base_dir}/scoring/reasoning/{safe_model_id4scoring}/{safe_model_id}"

    model, processor = load_model_n_prosessor(args, args.model_id, args.model_path)

    # If using ollama models, ensure they're available
    def ensure_ollama_model(model_name):
        import shutil, sys, subprocess
        if shutil.which('ollama') is None:
            print('Warning: ollama CLI not found in PATH. Ensure ollama is installed and daemon is running.')
            return
        model_short = model_name.split(':', 1)[1] if ':' in model_name else model_name
        try:
            res = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
            if model_short in res.stdout:
                return
            print(f"Ollama model '{model_short}' not found locally — attempting to pull...")
            pull = subprocess.run(["ollama", "pull", model_short], capture_output=True, text=True)
            if pull.returncode != 0:
                print(f"Failed to pull Ollama model '{model_short}': {pull.stderr}")
                print("Exiting due to failed model pull.")
                sys.exit(1)
        except Exception as e:
            print(f"Error checking/pulling Ollama model '{model_short}': {e}")
            sys.exit(1)

    ensure_ollama_model(args.model_id)
    ensure_ollama_model(args.model_id4scoring)

    with open(args.dx_by_dicoms_file, "r") as file:
        dx_by_dicoms = json.load(file)

    # Determine a dataset root for fallback lookups (may be empty)
    dataset_root_for_index = args.mimic_cxr_base or getattr(args, 'nih_cxr_base', None) or args.cxreasonbench_base_dir or ''

    # Build a filename->fullpath index for fast image lookup, but only for
    # targeted image folders to avoid walking the entire dataset root which
    # can be very large and block startup. Prefer indexing `pnt_base_dir`
    # and `segmask_base_dir` when available.
    image_index = {}
    index_roots = []
    if getattr(args, 'pnt_base_dir', None) and os.path.isdir(args.pnt_base_dir):
        index_roots.append(args.pnt_base_dir)
    if getattr(args, 'segmask_base_dir', None) and os.path.isdir(args.segmask_base_dir):
        index_roots.append(args.segmask_base_dir)

    for root in index_roots:
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                for fn in filenames:
                    if fn not in image_index:
                        image_index[fn] = os.path.join(dirpath, fn)
        except Exception:
            # best-effort: skip directories we cannot read
            pass

    def find_image_under_root(root, fname):
        # Prefer fast lookup via built index
        if fname in image_index:
            return image_index[fname]
        # If a specific root was provided, check direct join and shallow walk only
        if root and os.path.exists(root):
            cand = os.path.join(root, fname)
            if os.path.isfile(cand):
                return cand
            # shallow search: check immediate subdirs (avoid deep os.walk)
            try:
                for entry in os.scandir(root):
                    if entry.is_file() and entry.name == fname:
                        return entry.path
                    if entry.is_dir():
                        cand = os.path.join(entry.path, fname)
                        if os.path.isfile(cand):
                            return cand
            except Exception:
                pass
        return None

    for run_guidance_type in ['idk_init', 'idk_custom']:
        saved_dir_scoring_reasoning_list = glob(os.path.join(args.save_base_dir, 'scoring', 'reasoning',
                                     safe_model_id4scoring, safe_model_id, '*'))
        for saved_dir_scoring_reasoning in saved_dir_scoring_reasoning_list:
            target_dx = saved_dir_scoring_reasoning.split('/')[-1]

            correct_dicom_final = load_jsonl_if_exists(os.path.join(saved_dir_scoring_reasoning, "dicom_lst_correct_final.jsonl"))
            dicom_lst_idk_init = load_jsonl_if_exists(os.path.join(saved_dir_scoring_reasoning, "dicom_lst_idk_init.jsonl"))
            dicom_lst_idk_custom = load_jsonl_if_exists(os.path.join(saved_dir_scoring_reasoning, "dicom_lst_idk_custom.jsonl"))
            # By default guidance runs on idk lists. If forced, run for all dicoms that have reasoning outputs.
            if args.force_all_guidance:
                # collect dicoms that have reasoning outputs for this dx
                reasoning_dir = os.path.join(saved_dir_reasoning, target_dx)
                if os.path.isdir(reasoning_dir):
                    saved_files = [os.path.splitext(os.path.basename(p))[0] for p in glob(f"{reasoning_dir}/*.json")]
                    target_dicom_lst = saved_files
                else:
                    target_dicom_lst = []
            else:
                target_dicom_lst = dicom_lst_idk_init if 'idk_init' in run_guidance_type else dicom_lst_idk_custom

            if target_dicom_lst:
                save_dir_per_dx = os.path.join(args.save_base_dir, 'inference', 'guidance', safe_model_id, run_guidance_type, target_dx)
                save_dir_per_dx_scoring = os.path.join(args.save_base_dir, 'scoring', 'guidance', safe_model_id4scoring, safe_model_id, run_guidance_type, target_dx)

                os.makedirs(save_dir_per_dx, exist_ok=True)
                os.makedirs(save_dir_per_dx_scoring, exist_ok=True)

                saved_dicom_lst = [path.split('/')[-1].split('.')[0] for path in glob(f"{save_dir_per_dx}/*.json")]
                remained_dicom_lst = list(set(target_dicom_lst).difference(saved_dicom_lst))
                if args.test_limit is not None:
                    remained_dicom_lst = list(remained_dicom_lst)[:args.test_limit]
                for idxx, dicom in tqdm(enumerate(remained_dicom_lst), total=len(remained_dicom_lst), desc=f'{run_guidance_type}_{target_dx}_{args.model_id}'):
                    review_candidates = list(set(dx_by_dicoms[target_dx]).difference(correct_dicom_final))
                    if dicom in review_candidates:
                        review_candidates.remove(dicom)
                    if not review_candidates:
                        review_candidates = correct_dicom_final
                    # ====================================================================
                    #                       Prepare Chat History
                    # ====================================================================
                    responses_reasoning_fname = f"{saved_dir_reasoning}/{target_dx}/{dicom}.json"
                    with open(responses_reasoning_fname, "r") as file:
                        responses_reasoning = json.load(file)

                    system_message = responses_reasoning['system_message']

                    if 'healthgpt' in args.model_id.lower():
                        from llava import conversation as conversation_lib
                        conv = conversation_lib.conv_templates["phi4_instruct"].copy()

                    if run_guidance_type in ['idk_init']:
                        init_query = responses_reasoning['stage-init']['query']
                        init_response = responses_reasoning['stage-init']['response']
                        init_answer = responses_reasoning['stage-init']['answer']

                        qa_init_file = f'{args.qa_base_dir}/{target_dx}/path1/init/basic/{dicom}.json'
                        with open(qa_init_file, "r") as file:
                            qa_init = json.load(file)
                        # resolve image paths robustly
                        init_img_path = []
                        for path in qa_init.get('img_path', []):
                            if os.path.isabs(path) and os.path.exists(path):
                                init_img_path.append(path)
                            else:
                                cand = find_image_under_root(dataset_root_for_index, os.path.basename(path)) if dataset_root_for_index else None
                                if cand:
                                    init_img_path.append(cand)
                                else:
                                    init_img_path.append(os.path.join(args.mimic_cxr_base, path))
                        chat_history = [[init_query, init_img_path, init_response]]

                        if 'healthgpt' in args.model_id.lower():
                            conv.append_message(conv.roles[0], init_query)
                            conv.append_message(conv.roles[1], init_response)

                        answer_per_stage = {'reasoning-init': init_answer}

                    elif run_guidance_type in ['idk_custom']:
                        chat_history = []
                        answer_per_stage = {}
                        qa_init_file = f'{args.qa_base_dir}/{target_dx}/path1/init/basic/{dicom}.json'
                        with open(qa_init_file, "r") as file:
                            qa_init = json.load(file)
                        # resolve image paths robustly
                        init_img_path = []
                        for path in qa_init.get('img_path', []):
                            if os.path.isabs(path) and os.path.exists(path):
                                init_img_path.append(path)
                            else:
                                cand = find_image_under_root(dataset_root_for_index, os.path.basename(path)) if dataset_root_for_index else None
                                if cand:
                                    init_img_path.append(cand)
                                else:
                                    init_img_path.append(os.path.join(args.mimic_cxr_base, path))
                        for stage in responses_reasoning.keys():
                            if stage.startswith(('stage-init', 'stage-criteria', 'stage-custom_criteria')):
                                if stage.startswith('stage-init'):
                                    chat_history.append([responses_reasoning[stage]['query'],
                                                         init_img_path,
                                                         responses_reasoning[stage]['response']])
                                else:
                                    chat_history.append([responses_reasoning[stage]['query'],
                                                         responses_reasoning[stage]['img_path'],
                                                         responses_reasoning[stage]['response']])

                                if 'healthgpt' in args.model_id.lower():
                                    conv.append_message(conv.roles[0], responses_reasoning[stage]['query'])
                                    conv.append_message(conv.roles[1], responses_reasoning[stage]['response'])

                                answer_per_stage[f"reasoning-{stage.split('-')[-1]}"] = responses_reasoning[stage]['answer']

                    scoring_per_stage = {}
                    sysmsg4scoring_per_stage = {}
                    # ====================================================================
                    #                       Question - Guidance - Body Part
                    # ====================================================================
                    qa_bodypart_file = f'{args.qa_base_dir}/{target_dx}/path2/stage1/basic/{dicom}.json'
                    if not os.path.isfile(qa_bodypart_file):
                        print(f"Warning: missing QA bodypart file {qa_bodypart_file} for dicom {dicom}, skipping")
                        continue
                    with open(qa_bodypart_file, "r") as file:
                        qa_bodypart = json.load(file)

                    score_bodypart_lst = []

                    # normalize answers: qa_bodypart['answer'] may be a list (per-question)
                    # or a single value (applies to all questions). Use a safe accessor.
                    answers_bodypart = qa_bodypart.get('answer', [])

                    for idx, q_bodypart in enumerate(qa_bodypart['question']):
                        # Resolve segmask image paths robustly. qa_bodypart['img_path'] may contain
                        # relative paths, leading slashes, or directory paths. We must ensure we
                        # pass file paths (not directories) to the loader.
                        img_path_lst_bodypart = []
                        if idx == 0:
                            for path in qa_bodypart.get('img_path', []):
                                rel = path.lstrip('/\\')
                                cand = os.path.join(args.segmask_base_dir, rel) if getattr(args, 'segmask_base_dir', None) else None
                                if cand and os.path.isfile(cand):
                                    img_path_lst_bodypart.append(cand)
                                    continue
                                if cand and os.path.isdir(cand):
                                    # pick first file inside that directory
                                    found = None
                                    for root, _, files in os.walk(cand):
                                        for fn in files:
                                            found = os.path.join(root, fn)
                                            break
                                        if found:
                                            break
                                    if found:
                                        img_path_lst_bodypart.append(found)
                                        continue
                                # fallback: try to find by basename under segmask dx folder first (fast),
                                # then dataset root. This avoids walking the entire segmask base.
                                base = os.path.basename(rel)
                                found = None
                                # prefer searching only within segmask/<dx> if available
                                if getattr(args, 'segmask_base_dir', None):
                                    seg_dx_root = os.path.join(args.segmask_base_dir, target_dx)
                                    if os.path.isdir(seg_dx_root):
                                        found = find_image_under_root(seg_dx_root, base)
                                    else:
                                        found = find_image_under_root(args.segmask_base_dir, base)
                                # fallback to dataset root
                                if not found and dataset_root_for_index:
                                    found = find_image_under_root(dataset_root_for_index, base)
                                if found:
                                    img_path_lst_bodypart.append(found)
                        response_bodypart, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                         query=q_bodypart,
                                                                                         img_path_lst=img_path_lst_bodypart,
                                                                                         system_message=system_message,
                                                                                         chat_history=chat_history)
                        # select the corresponding answer safely
                        if isinstance(answers_bodypart, list):
                            ans_body = answers_bodypart[idx] if idx < len(answers_bodypart) else (answers_bodypart[-1] if answers_bodypart else '')
                        else:
                            ans_body = answers_bodypart

                        answer_per_stage[f'guidance-bodypart_{idx}'] = ans_body

                        score_bodypart_sub, res_idx_body_sub = return_scoring_result(args.model_id4scoring, stage_by_sysmsg['bodypart_one'],
                                                                                     q_bodypart,
                                                                                     ans_body,
                                                                                     response_bodypart)
                        score_bodypart_lst.append(score_bodypart_sub)
                        scoring_per_stage[f'guidance-bodypart_{idx}'] = score_bodypart_sub
                        sysmsg4scoring_per_stage[f'guidance-bodypart_{idx}'] = stage_by_sysmsg['bodypart_one']

                        if res_idx_body_sub is not None:
                            chat_history[-1][-1] = chat_history[-1][-1][res_idx_body_sub]

                    if len(score_bodypart_lst) == sum(score_bodypart_lst):
                        # ====================================================================
                        #                       Question - Guidance - Measurement
                        # ====================================================================
                        qa_measurement_file = f'{args.qa_base_dir}/{target_dx}/path2/stage2/basic/{dicom}.json'
                        with open(qa_measurement_file, "r") as file:
                            qa_measurement = json.load(file)

                        q_measurement = qa_measurement['question']
                        img_path_lst_measurement = [f'{args.pnt_base_dir}{path}' for path in qa_measurement['img_path']]
                        response_measurement, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                            query=q_measurement,
                                                                                            img_path_lst=img_path_lst_measurement,
                                                                                            system_message=system_message,
                                                                                            chat_history=chat_history)
                        answer_per_stage[f'guidance-measurement'] = qa_measurement['answer']

                        score_measurement, res_idx_measure = return_scoring_result(args.model_id4scoring, stage_by_sysmsg['measurement'],
                                                                                   qa_measurement['question'],
                                                                                   qa_measurement['answer'],
                                                                                   response_measurement)

                        if res_idx_measure is not None:
                            chat_history[-1][-1] = chat_history[-1][-1][res_idx_measure]

                        scoring_per_stage[f'guidance-measurement'] = score_measurement
                        sysmsg4scoring_per_stage[f'guidance-measurement'] = stage_by_sysmsg['measurement']

                        if score_measurement == 1:
                            # ====================================================================
                            #                       Question - Guidance - Final Diagnosis
                            # ====================================================================
                            qa_final_file = f'{args.qa_base_dir}/{target_dx}/path2/stage3/basic/{dicom}.json'

                            with open(qa_final_file, "r") as file:
                                qa_final = json.load(file)
                            measured_value = '' # qa_final['measured_value']
                            response_final, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                          query=qa_final['question'],
                                                                                          img_path_lst=[],
                                                                                          system_message=system_message,
                                                                                          chat_history=chat_history)
                            answer_per_stage['guidance-final'] = qa_final['answer']

                            score_final, res_idx_final = return_scoring_result(args.model_id4scoring, stage_by_sysmsg['final'],
                                                                               [qa_final['question']],
                                                                               qa_final['answer'], response_final)

                            if res_idx_final is not None:
                                chat_history[-1][-1] = chat_history[-1][-1][res_idx_final]
                                response_final = response_final[res_idx_final]

                            scoring_per_stage[f'guidance-final'] = score_final
                            sysmsg4scoring_per_stage[f'guidance-final'] = stage_by_sysmsg['final']


                            sysmsg_value_extraction = stage_by_sysmsg['extract_value_projection'] if target_dx in ['projection'] else stage_by_sysmsg['extract_value']
                            model_measured_value = return_measured_value_result(args.model_id4scoring, sysmsg_value_extraction, response_final) if target_dx in dx_task_measurement else ''

                            scoring_per_stage[f'measured_value_guidance'] = model_measured_value
                            sysmsg4scoring_per_stage[f'measured_value_guidance'] = sysmsg_value_extraction

                            if score_final == 1 and not getattr(args, 'no_review', False):
                                # ====================================================================
                                #                       Question - Review After Guidance - Init Diagnosis
                                # ====================================================================
                                review_dicom = random.choice(review_candidates)
                                qa_init_file = f'{args.qa_base_dir}/{target_dx}/re-path1/init/basic/{review_dicom}.json'
                                with open(qa_init_file, "r") as file:
                                    qa_init = json.load(file)
                                qa_init_img_path = []
                                for path in qa_init.get('img_path', []):
                                    if os.path.isabs(path) and os.path.exists(path):
                                        qa_init_img_path.append(path)
                                    else:
                                        cand = find_image_under_root(dataset_root_for_index, os.path.basename(path)) if dataset_root_for_index else None
                                        if cand:
                                            qa_init_img_path.append(cand)
                                        else:
                                            qa_init_img_path.append(os.path.join(args.mimic_cxr_base, path))
                                response_init, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                             query=qa_init['question'],
                                                                                             img_path_lst=qa_init_img_path,
                                                                                             system_message=system_message,
                                                                                             chat_history=chat_history)

                                answer_per_stage['review-init'] = qa_init['answer']

                                score_review_init, res_idx_r_init = return_scoring_result(args.model_id4scoring, stage_by_sysmsg['final'],
                                                                                          [qa_init['question']],
                                                                                          qa_init['answer'], response_init)

                                if res_idx_r_init is not None:
                                    chat_history[-1][-1] = chat_history[-1][-1][res_idx_r_init]

                                scoring_per_stage['review-init'] = score_review_init
                                sysmsg4scoring_per_stage['review-init'] = stage_by_sysmsg['final']

                                if score_review_init == 1:
                                    # ====================================================================
                                    #                       Question - Review - Criteria
                                    # ====================================================================
                                    qa_criteria_files = glob(f'{args.qa_base_dir}/{target_dx}/re-path1/stage1/*/{review_dicom}.json')
                                    if not qa_criteria_files:
                                        warn(f"Warning: QA criteria review file not found for review dicom {review_dicom}, dx {target_dx}. Skipping review")
                                        continue
                                    qa_criteria_file = qa_criteria_files[0]
                                    with open(qa_criteria_file, "r") as file:
                                        qa_criteria = json.load(file)

                                    score_review_criteria_lst = []
                                    for idx, (q_criteria, a_criteria) in enumerate(zip(qa_criteria['question'], qa_criteria['answer'])):
                                        response_review_criteria_sub, chat_history = inference_vllms(args.model_id)(args, model,
                                                                                                                    processor,
                                                                                                                    query=q_criteria,
                                                                                                                    img_path_lst=[],
                                                                                                                    system_message=system_message,
                                                                                                                    chat_history=chat_history)
                                        answer_per_stage[f'review-criteria_{idx}'] = a_criteria

                                        score_review_criteria_sub, res_idx_r_c = return_scoring_result(args.model_id4scoring, stage_by_sysmsg['criteria'],
                                                                                                       q_criteria, a_criteria, response_review_criteria_sub)
                                        score_review_criteria_lst.append(score_review_criteria_sub)
                                        scoring_per_stage[f'review-criteria_{idx}'] = score_review_criteria_sub
                                        sysmsg4scoring_per_stage[f'review-criteria_{idx}'] = stage_by_sysmsg['criteria']

                                        if res_idx_r_c is not None:
                                            chat_history[-1][-1] = chat_history[-1][-1][res_idx_r_c]

                                    if len(score_review_criteria_lst) == sum(score_review_criteria_lst):
                                        # ====================================================================
                                        #                       Question - Review - Body Part
                                        # ====================================================================
                                        qa_bodypart_files = glob(f'{args.qa_base_dir}/{target_dx}/re-path1/stage2/*/{review_dicom}.json')
                                        if not qa_bodypart_files:
                                            warn(f"Warning: QA bodypart review file not found for review dicom {review_dicom}, dx {target_dx}. Skipping review")
                                            continue
                                        qa_bodypart_file = qa_bodypart_files[0]
                                        with open(qa_bodypart_file, "r") as file:
                                            qa_bodypart = json.load(file)

                                        score_review_body_lst = []
                                        for idx, q_bodypart in enumerate(qa_bodypart['question']):
                                            img_path_lst_bodypart = [f'{args.segmask_base_dir}{path}' for path in qa_bodypart['img_path'][idx]]

                                            response_review_body, chat_history = inference_vllms(args.model_id)(args, model,
                                                                                                                processor,
                                                                                                                query=q_bodypart,
                                                                                                                img_path_lst=img_path_lst_bodypart,
                                                                                                                system_message=system_message,
                                                                                                                chat_history=chat_history)

                                            answers_bodypart_review = qa_bodypart.get('answer', [])
                                            if isinstance(answers_bodypart_review, list):
                                                ans_review = answers_bodypart_review[idx] if idx < len(answers_bodypart_review) else (answers_bodypart_review[-1] if answers_bodypart_review else '')
                                            else:
                                                ans_review = answers_bodypart_review

                                            answer_per_stage[f'review-bodypart_{idx}'] = ans_review

                                            sys_msg_bodypart = stage_by_sysmsg['bodypart_all'] if target_dx in dx_task_multi_bodyparts else stage_by_sysmsg['bodypart_one']

                                            score_review_body_sub, res_idx_r_b = return_scoring_result(args.model_id4scoring, sys_msg_bodypart,
                                                                                                       q_bodypart, ans_review,
                                                                                                       response_review_body)
                                            score_review_body_lst.append(score_review_body_sub)
                                            scoring_per_stage[f'review-bodypart_{idx}'] = score_review_body_sub
                                            sysmsg4scoring_per_stage[f'review-bodypart_{idx}'] = sys_msg_bodypart

                                            if res_idx_r_b is not None:
                                                chat_history[-1][-1] = chat_history[-1][-1][res_idx_r_b]

                                        if len(score_review_body_lst) == sum(score_review_body_lst):

                                            # ====================================================================
                                            #                       Question - Review - Measurement
                                            # ====================================================================
                                            qa_measurement_file = f'{args.qa_base_dir}/{target_dx}/re-path1/stage3/basic/{review_dicom}.json'
                                            with open(qa_measurement_file, "r") as file:
                                                qa_measurement = json.load(file)

                                            response_review_measure, chat_history = inference_vllms(args.model_id)(args, model,
                                                                                                                   processor,
                                                                                                                   query=qa_measurement['question'],
                                                                                                                   img_path_lst=[],
                                                                                                                   system_message=system_message,
                                                                                                                   chat_history=chat_history)
                                            answer_per_stage['review-measurement'] = qa_measurement['answer']

                                            # Use proper membership check for diagnostic tasks
                                            if target_dx in ['projection']:
                                                sys_msg_measurment = stage_by_sysmsg['measurement_projection']
                                            else:
                                                sys_msg_measurment = stage_by_sysmsg['measurement']

                                            score_review_measure, res_idx_r_measure = return_scoring_result(args.model_id4scoring, sys_msg_bodypart,
                                                                                                            qa_measurement['question'],
                                                                                                            qa_measurement['answer'],
                                                                                                            response_review_measure)

                                            if res_idx_r_measure is not None:
                                                chat_history[-1][-1] = chat_history[-1][-1][res_idx_r_measure]

                                            scoring_per_stage[f'review-measurement'] = score_review_measure
                                            sysmsg4scoring_per_stage[f'review-measurement'] = sys_msg_bodypart

                                            if score_review_measure == 1:
                                                # ====================================================================
                                                #                       Question - Review - Final
                                                # ====================================================================
                                                qa_final_file = f'{args.qa_base_dir}/{target_dx}/re-path1/stage4/basic/{review_dicom}.json'
                                                with open(qa_final_file, "r") as file:
                                                    qa_final = json.load(file)
                                                response_review_final, chat_history = inference_vllms(args.model_id)(args,
                                                                                                                     model,
                                                                                                                     processor,
                                                                                                                     query=qa_final['question'],
                                                                                                                     img_path_lst=[],
                                                                                                                     system_message=system_message,
                                                                                                                     chat_history=chat_history)
                                                answer_per_stage['review-final'] = qa_final['answer']

                                                score_review_final, res_idx_r_final = return_scoring_result(args.model_id4scoring, stage_by_sysmsg['final'],
                                                                                                            [qa_final['question']], qa_final['answer'], response_review_measure)
                                                if res_idx_r_final is not None:
                                                    chat_history[-1][-1] = chat_history[-1][-1][res_idx_r_final]
                                                    response_review_final = response_review_final[res_idx_r_final]

                                                scoring_per_stage[f'review-final'] = score_review_final
                                                sysmsg4scoring_per_stage[f'review-final'] = stage_by_sysmsg['final']

                                                sysmsg_value_extraction = stage_by_sysmsg['extract_value_projection'] if target_dx in ['projection'] else stage_by_sysmsg['extract_value']

                                                model_measured_value = return_measured_value_result(args.model_id4scoring, sysmsg_value_extraction, response_review_final) if target_dx in dx_task_measurement else ''

                                                scoring_per_stage[f'measured_value_review'] = model_measured_value
                                                sysmsg4scoring_per_stage[f'measured_value_review'] = sysmsg_value_extraction

                    # ====================================================================
                    #                       SAVE Result
                    # ====================================================================
                    result = {'dicom': dicom,
                              'system_message': system_message,
                              'cxr_path': init_img_path[0]}

                    if 'guidance-final' in scoring_per_stage:
                        result['measured_value'] = measured_value
                    if 'review-init' in scoring_per_stage:
                        result['dicom_review'] = review_dicom
                        result['cxr_path_review'] = qa_init_img_path[0]
                    if 'review-final' in scoring_per_stage:
                        result['measured_value_review'] = ''# qa_init['measured_value']

                    for stage, history in zip(answer_per_stage, chat_history):
                        hist = {f'stage-{stage}': {'query': history[0], 'img_path': history[1],
                                                   'response': history[-1], 'answer': answer_per_stage[stage]}}

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
