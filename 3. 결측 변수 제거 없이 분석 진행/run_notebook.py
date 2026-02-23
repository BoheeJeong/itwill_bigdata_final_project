# -*- coding: utf-8 -*-
"""Run a single ipynb and report first error."""
import json
import sys
import os
import matplotlib
matplotlib.use("Agg")

def run_notebook(path):
    os.chdir(os.path.dirname(os.path.abspath(path)))
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    cells = [c for c in nb['cells'] if c.get('cell_type') == 'code']
    globals_dict = {}
    for i, c in enumerate(cells):
        src = ''.join(c.get('source', []))
        if not src.strip():
            continue
        try:
            exec(compile(src, f'<cell {i}>', 'exec'), globals_dict)
        except Exception as e:
            print(f"ERROR in cell {i}: {e}")
            print("Code snippet:", src[:500])
            raise
    print("OK:", path)

if __name__ == '__main__':
    run_notebook(sys.argv[1])
