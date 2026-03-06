# -*- coding: utf-8 -*-
"""3번 폴더 노트북을 3-2 폴더로 복사하고 종속변수를 dependent_wage_work로 변경."""
import json
import os

SRC_DIR = r"3. 결측 변수 제거 없이 분석 진행"
DST_DIR = r"3-2. 결측 변수 제거 없이 분석 진행_임금근로"
NAMES = [
    "new_로지스틱",
    "new_경사하강법",
    "new_KNN",
    "new_SVM",
    "new_랜덤포레스트",
    "new_XGBoost",
    "new_LightGBM",
    "new_CatBoost",
]

for name in NAMES:
    src_path = os.path.join(SRC_DIR, name + ".ipynb")
    dst_path = os.path.join(DST_DIR, name + ".ipynb")
    if not os.path.isfile(src_path):
        print("Skip (no file):", src_path)
        continue
    with open(src_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []) if isinstance(cell["source"], list) else [cell["source"]])
        if cell.get("cell_type") == "markdown":
            if "3번" in src and ("결측" in src or "로지스틱" in src or "경사" in src):
                src = src.replace("결측 제거 없이 분석 (3번).", "결측 제거 없이 분석 (3-2). 종속변수: 임금근로(dependent_wage_work).")
                src = src.replace("(3번)", "(3-2, 임금근로)")
            cell["source"] = src.splitlines(keepends=True) if src else [src]
            continue
        if cell.get("cell_type") != "code":
            continue
        new_src = src
        new_src = new_src.replace("origin.drop(['dependent_wage_work'], axis=1)", "origin.drop(['dependent_ecotype'], axis=1)")
        new_src = new_src.replace('yname = "dependent_ecotype"', 'yname = "dependent_wage_work"')
        new_src = new_src.replace("yname = \"dependent_ecotype\"", "yname = \"dependent_wage_work\"")
        new_src = new_src.replace("3. 결측 변수 제거 없이 분석 진행", "3-2. 결측 변수 제거 없이 분석 진행_임금근로")
        if new_src != src:
            cell["source"] = new_src.splitlines(keepends=True) if new_src else [new_src]
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Written:", dst_path)
print("Done.")
