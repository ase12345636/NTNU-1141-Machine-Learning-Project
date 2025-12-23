from collections import Counter
import subprocess
import os
import time

# Prefer using local Ollama for scoring when model_id4scoring starts with 'ollama:'
try:
    from google import genai
    import google.genai.types as types
    _HAS_GOOGLE_GENAI = True
except Exception:
    _HAS_GOOGLE_GENAI = False


def _inference_ollama_scoring(model_id4scoring, system_message, prompt):
    model_short = model_id4scoring.split(':', 1)[1] if ':' in model_id4scoring else model_id4scoring
    full_prompt = f"{system_message}\n{prompt}" if system_message else prompt

    # best-effort debug directory for storing prompts/responses
    try:
        debug_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'result_ollama', 'debug')
        os.makedirs(debug_dir, exist_ok=True)
    except Exception:
        debug_dir = None

    # write prompt to debug file (truncated)
    try:
        if debug_dir:
            fn = f"scoring_prompt_{model_short}_{int(time.time())}.txt"
            with open(os.path.join(debug_dir, fn), 'w', encoding='utf-8') as _f:
                _f.write(full_prompt[:100000])
    except Exception:
        pass

    try:
        # Try several possible Ollama subcommands in case CLI version differs
        candidate_cmds = ["generate", "run", "eval", "predict", "chat"]
        last_proc = None
        for subcmd in candidate_cmds:
            try:
                proc = subprocess.run(["ollama", subcmd, model_short], input=full_prompt, capture_output=True, text=True, check=False, timeout=60)
            except subprocess.TimeoutExpired:
                # try next subcmd if timeout
                last_proc = None
                continue
            last_proc = proc
            if proc.returncode == 0:
                class _Out:
                    pass
                out = _Out()
                out.text = proc.stdout.strip()
                # write scoring response (best-effort)
                try:
                    if debug_dir:
                        ofn = f"scoring_response_{model_short}_{int(time.time())}.txt"
                        with open(os.path.join(debug_dir, ofn), 'w', encoding='utf-8') as _f:
                            _f.write(out.text[:200000])
                except Exception:
                    pass
                return out
            # if stderr indicates unknown command, try next
            if proc.stderr and "unknown command" in proc.stderr.lower():
                continue
            # otherwise break and return this proc's stderr
            break

        class _Out:
            pass
        out = _Out()
        if last_proc is not None:
            out.text = (last_proc.stderr or last_proc.stdout or "").strip()
        else:
            out.text = ""
        return out
    except FileNotFoundError:
        class _Out:
            pass
        out = _Out()
        out.text = ""
        return out
    except Exception as e:
        class _Out:
            pass
        out = _Out()
        out.text = f"[OLLAMA_ERROR] {e}"
        return out


def inference_llm4scoring(model_id4scoring, system_message, prompt):
    # Ollama local scoring
    if isinstance(model_id4scoring, str) and model_id4scoring.startswith('ollama:'):
        return _inference_ollama_scoring(model_id4scoring, system_message, prompt)

    # Google GenAI scoring (if available)
    if _HAS_GOOGLE_GENAI:
        client = genai.Client(http_options=types.HttpOptions(api_version="v1"))
        chat = client.chats.create(model=model_id4scoring,
                                   config=types.GenerateContentConfig(
                                       system_instruction=system_message,
                                       max_output_tokens=500,
                                       temperature=0.0
                                   ))
        response = chat.send_message(prompt)
        return response

    # Fallback: try Ollama without prefix
    try:
        return _inference_ollama_scoring(model_id4scoring, system_message, prompt)
    except Exception:
        class _Out:
            pass
        out = _Out()
        out.text = ""
        return out


def return_scoring_result(model_id4scoring, sysmsg, question, answer, response):
    if isinstance(response, list):
        score_lst = []
        for res in response:
            if isinstance(question, list):
                query = f"- Question: {question[0]}" \
                        f"- Answer 1: {answer} " \
                        f"- Answer 2: {res}"
            else:
                query = f"- Answer 1: {answer} " \
                        f"- Answer 2: {res}"
            output = inference_llm4scoring(model_id4scoring, sysmsg, query)
            if 'true' in output.text.lower():
                score = 1
            elif 'idk' in output.text.lower():
                score = -1
            elif 'n/a' in output.text.lower():
                score = -2
            else:
                score = 0

            score_lst.append(score)
        most_common = Counter(score_lst).most_common()
        if len(most_common) < 2 or most_common[0][1] > most_common[1][1]:
            major_score = most_common[0][0]
        else:
            if -1 in score_lst:
                major_score = -1
            else:
                major_score = 0
        return major_score, score_lst.index(major_score)

    else:
        if isinstance(question, list):
            query = f"- Question: {question[0]}" \
                    f"- Answer 1: {answer} " \
                    f"- Answer 2: {response}"
        else:
            query = f"- Answer 1: {answer} " \
                    f"- Answer 2: {response}"
        output = inference_llm4scoring(model_id4scoring, sysmsg, query)

        if 'true' in output.text.lower():
            score = 1
        elif 'idk' in output.text.lower():
            score = -1
        elif 'n/a' in output.text.lower():
            score = -2
        else:
            score = 0

        return score, None


def return_measured_value_result(model_id4scoring, sysmsg, response):
    query = f"- Response: {response}"
    output = inference_llm4scoring(model_id4scoring, sysmsg, query)
    return output.text