# -*- coding: utf-8 -*-
"""Run all .ipynb in folders 1~10 in order. On first failure, print error and exit.
일시정지 후 다시 실행: python run_all_ipynb.py --from-folder "4. 결측 50% 초과 변수 제거 분석"
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# 1~2번은 raw 데이터(Lt09.csv, w09.csv 등) 필요. 없으면 스킵하려면 아래에서 start_from 적용.
FOLDER_ORDER = [
    "1. 초기 데이터 전처리",
    "2. 데이터 품질 점검",
    "3. 결측 변수 제거 없이 분석 진행",
    "4. 결측 50% 초과 변수 제거 분석",
    "5. 결측 80% 초과 변수 제거 분석",
    "6. 로그변환_결측제거없음",
    "7. 로그변환_결측50초과제거",
    "8. 로그변환_결측80초과제거",
    "9. 성별별 비교",
    "10. 배우자 유무별 비교",
]

SKIP_NAMES = ("1.파이캐럿 실패", "참고")  # 제목에 포함되면 스킵

def list_ipynb(folder_path):
    seen = set()
    out = []
    for name in sorted(os.listdir(folder_path)):
        if not name.endswith(".ipynb"):
            continue
        full = os.path.join(folder_path, name)
        key = os.path.normpath(full)
        if key in seen:
            continue
        if any(skip in name for skip in SKIP_NAMES):
            print(f"  (skip) {name}")
            continue
        seen.add(key)
        out.append(full)
    return out

def main():
    parser = argparse.ArgumentParser(description="Run all ipynb in folders 1~10. Use --from-folder to resume.")
    parser.add_argument(
        "--from-folder",
        type=str,
        default=None,
        help='Resume from this folder (e.g. "4. 결측 50% 초과 변수 제거 분석" or "4" or "6. 로그변환_결측제거없음")',
    )
    args = parser.parse_args()

    # 3번부터 실행 (1~2번은 raw 데이터 필요)
    start_from_index = 2
    folders = FOLDER_ORDER[start_from_index:]

    # --from-folder 이 있으면 해당 폴더부터 실행
    if args.from_folder:
        for i, fn in enumerate(FOLDER_ORDER):
            if args.from_folder in fn or fn.startswith(args.from_folder + "."):
                folders = FOLDER_ORDER[i:]
                print(f"(Resume) from folder: {folders[0]}")
                break
        else:
            print(f"No folder matching '{args.from_folder}'. Running from 3번.", file=sys.stderr)

    total = 0
    for folder_name in folders:
        folder_path = os.path.join(ROOT, folder_name)
        if not os.path.isdir(folder_path):
            continue
        notebooks = list_ipynb(folder_path)
        for path in notebooks:
            total += 1
            rel = os.path.relpath(path, ROOT)
            print(f"[{total}] Running: {rel}")
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
    print("All notebooks completed.")

if __name__ == "__main__":
    main()
