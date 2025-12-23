"""
Fix Value formatting in generated QA JSON files.
Finds answers containing 'Value:' or 'Value: Right -' and ensures numeric values are wrapped in brackets.
Run:
    python tools/fix_value_brackets.py --base d:/CXReasonBench/output_nih --inference_path path2
"""
import os
import re
import json
import argparse

VALUE_SIMPLE_RE = re.compile(r"Value:\s*([0-9]+(?:\.[0-9]+)?)")
VALUE_RIGHT_LEFT_RE = re.compile(r"Value:\s*Right\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*Left\s*-\s*([0-9]+(?:\.[0-9]+)?)")


def fix_text(s):
    if not isinstance(s, str):
        return s
    # Right/Left first
    s2 = VALUE_RIGHT_LEFT_RE.sub(r"Value: Right - [\1], Left - [\2]", s)
    s2 = VALUE_SIMPLE_RE.sub(r"Value: [\1]", s2)
    return s2


def process_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print('skip (read error)', path, e)
        return False
    changed = False
    # fix answer field if present
    if 'answer' in data:
        ans = data['answer']
        if isinstance(ans, str):
            new = fix_text(ans)
            if new != ans:
                data['answer'] = new
                changed = True
        elif isinstance(ans, list):
            new_list = []
            for a in ans:
                if isinstance(a, str):
                    new_list.append(fix_text(a))
                else:
                    new_list.append(a)
            if new_list != ans:
                data['answer'] = new_list
                changed = True
    # also check any nested fields with 'answer' keys
    # write back
    if changed:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print('fixed', path)
            return True
        except Exception as e:
            print('skip (write error)', path, e)
            return False
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--inference_path', default='path2')
    args = parser.parse_args()

    base = args.base
    qa_root = os.path.join(base, 'qa')
    count = 0
    fixed = 0
    for dx in os.listdir(qa_root):
        dx_dir = os.path.join(qa_root, dx, args.inference_path)
        if not os.path.isdir(dx_dir):
            continue
        # look under stage3/basic and final/basic as fallback
        candidates = [os.path.join(dx_dir, 'stage3', 'basic'), os.path.join(dx_dir, 'final', 'basic')]
        for cand in candidates:
            if not os.path.isdir(cand):
                continue
            for fn in os.listdir(cand):
                if not fn.endswith('.json'):
                    continue
                path = os.path.join(cand, fn)
                count += 1
                if process_file(path):
                    fixed += 1
    print(f"scanned={count} fixed={fixed}")

if __name__ == '__main__':
    main()
