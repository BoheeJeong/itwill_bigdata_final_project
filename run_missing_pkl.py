# -*- coding: utf-8 -*-
"""pkl이 없는 모델 노트북만 실행 (3~8번 폴더의 new_*.ipynb → results/*.pkl)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# pkl을 만드는 폴더만 (3~8번)
FOLDER_ORDER = [
    "3. 결측 변수 제거 없이 분석 진행",
    "4. 결측 50% 초과 변수 제거 분석",
    "5. 결측 80% 초과 변수 제거 분석",
    "6. 로그변환_결측제거없음",
    "7. 로그변환_결측50초과제거",
    "8. 로그변환_결측80초과제거",
]

def main():
    to_run = []
    for folder_name in FOLDER_ORDER:
        folder_path = os.path.join(ROOT, folder_name)
        results_dir = os.path.join(folder_path, "results")
        if not os.path.isdir(folder_path):
            continue
        for name in sorted(os.listdir(folder_path)):
            if not name.endswith(".ipynb") or not name.startswith("new_"):
                continue
            if "종합" in name:
                continue
            stem = name[:-6]  # .ipynb 제거
            pkl_path = os.path.join(results_dir, stem + ".pkl")
            if os.path.isfile(pkl_path):
                continue
            full = os.path.join(folder_path, name)
            to_run.append(full)

    if not to_run:
        print("pkl 없는 모델 없음. 모두 완료.")
        return

    print(f"pkl 없는 노트북 {len(to_run)}개 실행합니다.")
    for i, path in enumerate(to_run, 1):
        rel = os.path.relpath(path, ROOT)
        print(f"[{i}/{len(to_run)}] Running: {rel}")
        sys.stdout.flush()
        r = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--execute", "--to", "notebook", "--inplace",
                "--ExecutePreprocessor.timeout=1800",
                path,
            ],
            cwd=ROOT,
        )
        if r.returncode != 0:
            print(f"FAILED: {rel}", file=sys.stderr)
            sys.exit(1)
    print("All missing notebooks completed.")

if __name__ == "__main__":
    main()
