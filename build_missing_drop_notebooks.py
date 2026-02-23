# -*- coding: utf-8 -*-
"""
결측 50% 초과 / 80% 초과 변수 제거 버전 노트북 생성.
실행: python build_missing_drop_notebooks.py
"""
import os
import json
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FOLDER_3 = os.path.join(PROJECT_ROOT, "3. 결측 변수 제거 없이 분석 진행")
FOLDER_50 = os.path.join(PROJECT_ROOT, "4. 결측 50% 초과 변수 제거 분석")
FOLDER_80 = os.path.join(PROJECT_ROOT, "5. 결측 80% 초과 변수 제거 분석")

# 모델별 파일명 (로지스틱 제외한 7개는 기존 new_* 에서 복사)
MODEL_NOTEBOOKS = [
    "new_경사하강법", "new_KNN", "new_SVM", "new_랜덤포레스트",
    "new_XGBoost", "new_LightGBM", "new_CatBoost"
]
PKL_NAMES = ["new_로지스틱"] + MODEL_NOTEBOOKS


def clear_outputs(nb):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return nb


def add_sys_path_to_source(source_list, insert_path):
    """analysis_utils import 셀 앞에 sys.path 추가."""
    joined = "".join(s if isinstance(s, str) else "" for s in source_list)
    if "from analysis_utils import" not in joined:
        return source_list
    new_lines = [
        "import sys\n",
        "sys.path.insert(0, r'" + insert_path + "')\n",
        "\n",
    ]
    return new_lines + source_list


def replace_data_loading_with_missing_drop(source_list, threshold_pct):
    """데이터 로딩 셀을 결측 비율 제거 버전으로 교체. threshold_pct: 0.5 or 0.8"""
    joined = "".join(s if isinstance(s, str) else "" for s in source_list)
    if "origin2 = origin.drop(['dependent_wage_work']" not in joined and "drop_for_leakage" not in joined:
        return source_list
    # 기존: origin2 = origin.drop(...), df2 = origin2.copy(), df3 = df2.copy(), yname, drop_for_leakage, x = df3.drop(...), y = ..., train_test_split
    # 새로: origin2 = ..., 결측률 계산 → 제거할 컬럼 → origin3 = origin2.drop(columns=drop_high_missing), df2 = origin3.copy(), df3 = df2.copy(), ...
    threshold = threshold_pct  # 0.5 or 0.8
    new_source = [
        "origin2 = origin.drop(['dependent_wage_work'], axis=1)\n",
        "yname = \"dependent_ecotype\"\n",
        "# 결측치가 {}% 초과인 변수 제거 (타깃 제외)\n".format(int(threshold * 100)),
        "missing_rate = origin2.isnull().mean()\n",
        "drop_high_missing = [c for c in missing_rate[missing_rate > {}].index if c != yname]\n".format(threshold),
        "origin3 = origin2.drop(columns=drop_high_missing)\n",
        "print(f'결측 {}% 초과 변수 제거: {{len(drop_high_missing)}}개 제거, 남은 컬럼 {{origin3.shape[1]}}개')\n".format(int(threshold * 100)),
        "df2 = origin3.copy()\n",
        "df3 = df2.copy()\n",
        "\n",
        "drop_for_leakage = [yname, 'work_ability_age']\n",
        "x = df3.drop(columns=[c for c in drop_for_leakage if c in df3.columns])\n",
        "y = df3[yname].astype(int)\n",
        "\n",
        "x_train, x_test, y_train, y_test = train_test_split(\n",
        "    x, y, test_size=0.25, random_state=52, stratify=y\n",
        ")\n",
        "x_train.shape, x_test.shape, y_train.shape, y_test.shape\n",
    ]
    return new_source


def replace_data_loading_cell(nb, threshold_pct):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        joined = "".join(s if isinstance(s, str) else "" for s in src)
        if "origin2 = origin.drop" in joined and "train_test_split" in joined and "x_train" in joined:
            cell["source"] = replace_data_loading_with_missing_drop(src, threshold_pct)
            return nb
    return nb


def add_sys_path_to_notebook(nb, folder_3_path):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        joined = "".join(s if isinstance(s, str) else "" for s in src)
        if "from analysis_utils import" in joined and "sys.path" not in joined:
            cell["source"] = add_sys_path_to_source(src, folder_3_path)
            return nb
    return nb


def update_title(nb, title_suffix):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown" and cell.get("source"):
            first = "".join(cell["source"])[:80]
            if "분석 (독립 실행용)" in first or "독립 실행용" in first:
                # 첫 줄만 수정 (서브타이틀 추가)
                lines = cell["source"]
                if isinstance(lines[0], str) and "\n" not in lines[0]:
                    cell["source"] = [lines[0].rstrip() + " " + title_suffix + "\n"] + (lines[1:] if len(lines) > 1 else [])
                break
    return nb


def ensure_csv_path(nb, project_root):
    """read_csv 경로를 프로젝트 루트 기준으로 설정."""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        for i, line in enumerate(src):
            if isinstance(line, str) and "read_csv" in line and "coding_book_mapping" in line:
                # 상대 경로로 1. 초기 데이터 전처리 사용
                base = os.path.join(project_root, "1. 초기 데이터 전처리", "3.coding_book_mapping.csv")
                src[i] = "origin = read_csv(r'" + base.replace("\\", "\\\\") + "', encoding='utf-8')\n"
                return nb
    return nb


def create_folder_and_copy_utils(folder_path, folder_3):
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(os.path.join(folder_path, "results"), exist_ok=True)
    # analysis_utils는 3번 폴더 것을 쓰므로 sys.path로 처리 (노트북에서 추가)


def build_one_notebook(src_path, dest_path, threshold_pct, folder_3_path, project_root, title_suffix):
    with open(src_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    nb = clear_outputs(nb)
    nb = add_sys_path_to_notebook(nb, folder_3_path)
    nb = replace_data_loading_cell(nb, threshold_pct)
    nb = update_title(nb, title_suffix)
    nb = ensure_csv_path(nb, project_root)
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print("  wrote", os.path.basename(dest_path))


def make_logistic_from_sgd(folder_3_path, project_root):
    """new_경사하강법을 복사해 로지스틱 회귀 버전 생성 (모델/param_grid만 변경)."""
    path = os.path.join(folder_3_path, "new_경사하강법.ipynb")
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    nb = clear_outputs(nb)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        raw = "".join(s if isinstance(s, str) else "" for s in src)
        if "from analysis_utils import" in raw:
            cell["source"] = add_sys_path_to_source(src, folder_3_path)
        if "SGDClassifier" in raw:
            full = "".join(s if isinstance(s, str) else "" for s in src)
            full = full.replace("from sklearn.linear_model import SGDClassifier", "from sklearn.linear_model import LogisticRegression")
            full = full.replace("SGDClassifier(random_state=52, loss='log_loss', max_iter=2000)", "LogisticRegression(random_state=52, max_iter=1000)")
            # param_grid 블록 전체 교체 (경사하강법 -> 로지스틱)
            import re
            old_grid = re.search(r'param_grid\s*=\s*\{[^}]+\}', full, re.DOTALL)
            if old_grid:
                new_grid = """param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__max_iter": [500, 1000],
    "model__class_weight": [None, "balanced"],
    "model__solver": ["lbfgs"],
    "model__penalty": ["l2"],
}"""
                full = full[:old_grid.start()] + new_grid + full[old_grid.end():]
            cell["source"] = [full]
    return nb


def main():
    os.makedirs(FOLDER_50, exist_ok=True)
    os.makedirs(FOLDER_80, exist_ok=True)
    os.makedirs(os.path.join(FOLDER_50, "results"), exist_ok=True)
    os.makedirs(os.path.join(FOLDER_80, "results"), exist_ok=True)

    for folder_path, pct, suffix in [
        (FOLDER_50, 0.5, "(결측 50% 초과 변수 제거)"),
        (FOLDER_80, 0.8, "(결측 80% 초과 변수 제거)"),
    ]:
        print("\n---", folder_path, "---")
        # 1) 로지스틱 노트북 생성 (경사하강법 기반)
        logistic_nb = make_logistic_from_sgd(FOLDER_3, PROJECT_ROOT)
        logistic_nb = add_sys_path_to_notebook(logistic_nb, FOLDER_3)
        logistic_nb = replace_data_loading_cell(logistic_nb, pct)
        logistic_nb = update_title(logistic_nb, suffix)
        # 제목을 로지스틱으로
        for c in logistic_nb.get("cells", []):
            if c.get("cell_type") == "markdown" and c.get("source"):
                txt = "".join(c["source"])
                if "경사하강법" in txt and "독립" in txt:
                    c["source"] = ["# 로지스틱 회귀 분석 (독립 실행용)\n", "\n", "결측 {}% 초과 변수 제거 버전.\n".format(int(pct * 100))]
                    break
        ensure_csv_path(logistic_nb, PROJECT_ROOT)
        dest = os.path.join(folder_path, "new_로지스틱.ipynb")
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(logistic_nb, f, ensure_ascii=False, indent=2)
        print("  wrote new_로지스틱.ipynb")

        # 2) 나머지 7개 모델 노트북
        for name in MODEL_NOTEBOOKS:
            src_path = os.path.join(FOLDER_3, name + ".ipynb")
            if not os.path.isfile(src_path):
                print("  skip (not found):", name)
                continue
            dest_path = os.path.join(folder_path, name + ".ipynb")
            build_one_notebook(src_path, dest_path, pct, FOLDER_3, PROJECT_ROOT, suffix)

    # 종합.ipynb (50%, 80% 각각)
    for folder_path, pct in [(FOLDER_50, 50), (FOLDER_80, 80)]:
        summary = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# 비선형 분류 모형 종합 비교 (결측 {}% 초과 변수 제거)\n".format(pct),
                        "\n",
                        "로지스틱, 경사하강법, KNN, SVM, 랜덤포레스트, XGBoost, LightGBM, CatBoost 결과를 불러와 **주요 성능 비교**, **과적합 여부**, **채택 모형 SHAP**을 한눈에 봅니다.\n",
                        "\n",
                        "**사전 조건:** 각 `new_*.ipynb`를 실행해 `results/*.pkl`이 생성되어 있어야 합니다."
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## 라이브러리 및 결과 로드"]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import os\n",
                        "import pickle\n",
                        "import numpy as np\n",
                        "import pandas as pd\n",
                        "from pandas import DataFrame\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sb\n",
                        "import shap\n",
                        "\n",
                        "my_dpi = 100\n",
                        "\n",
                        "results_dir = 'results'\n",
                        "pkl_names = [\n",
                        "    'new_로지스틱', 'new_경사하강법', 'new_KNN', 'new_SVM', 'new_랜덤포레스트',\n",
                        "    'new_XGBoost', 'new_LightGBM', 'new_CatBoost'\n",
                        "]\n",
                        "\n",
                        "results = {}\n",
                        "for name in pkl_names:\n",
                        "    path = os.path.join(results_dir, name + '.pkl')\n",
                        "    if os.path.isfile(path):\n",
                        "        with open(path, 'rb') as f:\n",
                        "            results[name] = pickle.load(f)\n",
                        "        print(f'Loaded: {name}')\n",
                        "    else:\n",
                        "        print(f'Skip (not found): {path}')\n",
                        "\n",
                        "if not results:\n",
                        "    raise SystemExit('결과 파일이 없습니다. 각 new_*.ipynb를 먼저 실행하세요.')"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## 1. 주요 성능 비교표"]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "score_dfs = {k: v.get('score_df') for k, v in results.items() if v.get('score_df') is not None}\n",
                        "if score_dfs:\n",
                        "    summary_scores = pd.concat(score_dfs, axis=0)\n",
                        "    display(summary_scores)\n",
                        "else:\n",
                        "    print('score_df 없음')"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## 2. 성능 비교 막대 그래프 (AUC 기준)"]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "if score_dfs:\n",
                        "    names = list(results.keys())\n",
                        "    aucs = []\n",
                        "    for n in names:\n",
                        "        df = results[n].get('score_df')\n",
                        "        if df is not None and not df.empty:\n",
                        "            a = df.get('AUC', df.get('roc_auc', pd.Series([None]))).iloc[0]\n",
                        "            aucs.append(a)\n",
                        "        else:\n",
                        "            aucs.append(None)\n",
                        "    fig, ax = plt.subplots(figsize=(10, 5), dpi=my_dpi)\n",
                        "    ax.barh(names, aucs)\n",
                        "    ax.set_xlabel('AUC')\n",
                        "    ax.set_title('모델별 AUC (결측 {}% 초과 제거)')\n".format(pct),
                        "    plt.tight_layout()\n",
                        "    plt.show()\n",
                        "else:\n",
                        "    print('score_df 없음')"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## 3. 과적합 여부 요약"]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "for name, data in results.items():\n",
                        "    status = data.get('overfit_status', 'N/A')\n",
                        "    print(f'{name}: {status}')"
                    ]
                },
            ],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11.0"}},
            "nbformat": 4,
            "nbformat_minor": 4,
        }
        dest = os.path.join(folder_path, "종합.ipynb")
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("  wrote 종합.ipynb ->", folder_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
