import os
import subprocess
import time
from datetime import datetime

ROOT = os.getcwd()
DEBUG_DIR = os.path.join(ROOT, 'result_ollama', 'debug')
os.makedirs(DEBUG_DIR, exist_ok=True)

# Minimal 1x1 PNG base64 data (transparent)
IMG_DATA = (
    'data:image/png;base64,'
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVQYV2NgYAAAAAMAAWgmWQ0'
    'AAAAASUVORK5CYII='
)

PROMPT = (
    "[PROBE] This is an automated multimodal probe. Below is an embedded image data URL.\n"
    "Please reply with one short line indicating whether you can see or process the image.\n\n"
    f"{IMG_DATA}\n\n"
    "Answer:"
)

MODEL_CANDIDATES = ['ollama:llama3.2-vision', 'llama3.2-vision']
CMD_CANDIDATES = ['run', 'chat', 'generate']


def save(path, data, mode='w'):
    with open(path, mode, encoding='utf-8', errors='ignore') as f:
        f.write(data)


def ts():
    return datetime.now().strftime('%Y%m%dT%H%M%S')


def try_cmd(cmd, input_bytes=None, timeout=30):
    try:
        proc = subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return proc.returncode, proc.stdout.decode('utf-8', errors='ignore'), proc.stderr.decode('utf-8', errors='ignore')
    except FileNotFoundError as e:
        return None, '', f'FileNotFoundError: {e}'
    except subprocess.TimeoutExpired as e:
        return -2, '', f'TimeoutExpired: {e}'


def main():
    start = ts()
    summary = []

    # Basic CLI checks
    rc, out, err = try_cmd(['ollama', 'version'])
    save(os.path.join(DEBUG_DIR, f'probe_ollama_version_{start}.txt'), out + '\n' + err)
    summary.append(('version', rc))

    rc, out, err = try_cmd(['ollama', 'list'])
    save(os.path.join(DEBUG_DIR, f'probe_ollama_list_{start}.txt'), out + '\n' + err)
    summary.append(('list', rc))

    prompt_bytes = PROMPT.encode('utf-8')
    save(os.path.join(DEBUG_DIR, f'probe_prompt_{start}.txt'), PROMPT)

    for cmd in CMD_CANDIDATES:
        for model in MODEL_CANDIDATES:
            name = f'probe_{cmd}_{model.replace(":","_")}_{start}.txt'
            stdout_path = os.path.join(DEBUG_DIR, 'stdout_' + name)
            stderr_path = os.path.join(DEBUG_DIR, 'stderr_' + name)
            print(f'Trying: ollama {cmd} {model} ...')
            rc, out, err = try_cmd(['ollama', cmd, model], input_bytes=prompt_bytes, timeout=30)
            if rc is None:
                print('ollama CLI not found or command not available')
            else:
                print(f'returncode={rc}')
            save(stdout_path, out)
            save(stderr_path, err)
            summary.append((f'{cmd} {model}', rc))

    # summary
    sum_path = os.path.join(DEBUG_DIR, f'probe_summary_{start}.txt')
    s = '\n'.join([f'{k}: {v}' for k, v in summary])
    save(sum_path, s)
    print('\nProbe complete. Debug files written to:', DEBUG_DIR)


if __name__ == '__main__':
    main()