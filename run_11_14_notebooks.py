# 11~14번 폴더의 모든 new_*.ipynb 실행. 오류 나면 출력 후 다음 진행.
import os
import subprocess
import sys

PROJECT_ROOT = r"c:\itwill_bigdata_final_project-main\itwill_bigdata_final_project"
FOLDERS = [
    "11. 연속형결측_평균대체",
    "12. 연속형결측_KNN대체",
    "13. 연속형결측_Iterative대체",
    "14. 연속형결측_median+결측플래그",
]
NOTEBOOKS = [
    "new_로지스틱",
    "new_경사하강법",
    "new_KNN",
    "new_SVM",
    "new_랜덤포레스트",
    "new_XGBoost",
    "new_LightGBM",
    "new_CatBoost",
]

timeout_sec = 1200  # SVM/LightGBM 등 최대 20분

for folder in FOLDERS:
    folder_path = os.path.join(PROJECT_ROOT, folder)
    for nb in NOTEBOOKS:
        path = os.path.join(folder_path, nb + ".ipynb")
        if not os.path.isfile(path):
            print(f"SKIP (no file): {folder} / {nb}")
            continue
        print(f"Run: {folder} / {nb} ...", flush=True)
        try:
            r = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--inplace",
                    path,
                    f"--ExecutePreprocessor.timeout={timeout_sec}",
                ],
                cwd=folder_path,
                capture_output=True,
                text=True,
                timeout=timeout_sec + 60,
            )
            if r.returncode != 0:
                print(f"  FAILED: {r.stderr[:500] if r.stderr else r.stdout[-500:]}")
            else:
                print(f"  OK")
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT")
        except Exception as e:
            print(f"  ERROR: {e}")

print("Done.")
