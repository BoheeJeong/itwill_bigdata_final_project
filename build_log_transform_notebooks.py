# -*- coding: utf-8 -*-
"""
로그 변환 적용 버전 노트북 3종 생성:
1) 결측치 제거 없음 + 로그변환
2) 결측 50% 초과 변수 제거 + 로그변환
3) 결측 80% 초과 변수 제거 + 로그변환
실행: python build_log_transform_notebooks.py
"""
import os
import json
import re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FOLDER_3 = os.path.join(PROJECT_ROOT, "3. 결측 변수 제거 없이 분석 진행")
FOLDER_50 = os.path.join(PROJECT_ROOT, "4. 결측 50% 초과 변수 제거 분석")
FOLDER_80 = os.path.join(PROJECT_ROOT, "5. 결측 80% 초과 변수 제거 분석")
FOLDER_LOG_NO = os.path.join(PROJECT_ROOT, "6. 로그변환_결측제거없음")
FOLDER_LOG_50 = os.path.join(PROJECT_ROOT, "7. 로그변환_결측50초과제거")
FOLDER_LOG_80 = os.path.join(PROJECT_ROOT, "8. 로그변환_결측80초과제거")

LOG_COLS_CANDIDATES = [
    "w09earned", "w09pinc", "w09e201", "w09e207", "w09e213", "w09e219",
    "w09e225", "w09e231", "w09e237", "w09e243", "w09e273", "w09e251",
    "w09passets", "w09pliabilities", "w09pnetassets", "w09hhinc",
    "w09hhassets", "w09hhliabilities", "w09hhnetassets",
    "w09fromchildren", "w09tochildren", "w09transferfrom", "w09transferto",
]

MODEL_NOTEBOOKS = [
    "new_경사하강법", "new_KNN", "new_SVM", "new_랜덤포레스트",
    "new_XGBoost", "new_LightGBM", "new_CatBoost",
]


def clear_outputs(nb):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return nb


def add_sys_path_to_source(source_list, insert_path):
    joined = "".join(s if isinstance(s, str) else "" for s in source_list)
    if "from analysis_utils import" not in joined:
        return source_list
    if "sys.path" in joined:
        return source_list
    new_lines = ["import sys\n", "sys.path.insert(0, r'" + insert_path + "')\n", "\n"]
    return new_lines + source_list


def add_sys_path_to_notebook(nb, folder_3_path):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        joined = "".join(s if isinstance(s, str) else "" for s in src)
        if "from analysis_utils import" in joined:
            cell["source"] = add_sys_path_to_source(src, folder_3_path)
            return nb
    return nb


def add_function_transformer_import(nb):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        joined = "".join(s if isinstance(s, str) else "" for s in src)
        if "from sklearn.preprocessing import StandardScaler" in joined and "FunctionTransformer" not in joined:
            for i, line in enumerate(src):
                if isinstance(line, str) and "from sklearn.preprocessing import StandardScaler" in line:
                    src[i] = line.replace(
                        "from sklearn.preprocessing import StandardScaler",
                        "from sklearn.preprocessing import StandardScaler, FunctionTransformer"
                    )
                    return nb
    return nb


def replace_pipeline_with_log_version(nb):
    """numeric_pipe + ColumnTransformer 셀을 로그 변환 적용 버전으로 교체."""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        joined = "".join(s if isinstance(s, str) else "" for s in src)
        if "numeric_pipe = Pipeline" not in joined or "ColumnTransformer" not in joined:
            continue
        # 기존: numeric_pipe, categorical_pipe, preprocess = ColumnTransformer( ("num", numeric_pipe, num_cols), ("cat", ...) )
        # 신규: log_cols, other_num_cols, num_log_pipe, num_other_pipe, categorical_pipe, preprocess (조건부 transformers)
        new_source = [
            "# 로그 변환 적용: 금액/자산/소득 등 연속형 일부에 log1p 적용\n",
            "LOG_COLS_CANDIDATES = " + str(LOG_COLS_CANDIDATES) + "\n",
            "log_cols = [c for c in LOG_COLS_CANDIDATES if c in num_cols.tolist()]\n",
            "other_num_cols = [c for c in num_cols if c not in log_cols]\n",
            "\n",
            "num_log_pipe = Pipeline([\n",
            "    (\"imputer\", SimpleImputer(strategy=\"median\")),\n",
            "    (\"log\", FunctionTransformer(np.log1p)),\n",
            "    (\"scaler\", StandardScaler())\n",
            "])\n",
            "\n",
            "num_other_pipe = Pipeline([\n",
            "    (\"imputer\", SimpleImputer(strategy=\"median\")),\n",
            "    (\"scaler\", StandardScaler())\n",
            "])\n",
            "\n",
            "categorical_pipe = Pipeline([\n",
            "    (\"imputer\", SimpleImputer(strategy=\"constant\", fill_value=\"Missing\")),\n",
            "    (\"onehot\", OneHotEncoder(drop=\"first\", handle_unknown=\"ignore\"))\n",
            "])\n",
            "\n",
            "transformers_list = []\n",
            "if len(log_cols) > 0:\n",
            "    transformers_list.append((\"num_log\", num_log_pipe, log_cols))\n",
            "if len(other_num_cols) > 0:\n",
            "    transformers_list.append((\"num_other\", num_other_pipe, other_num_cols))\n",
            "transformers_list.append((\"cat\", categorical_pipe, cat_cols))\n",
            "\n",
            "preprocess = ColumnTransformer(transformers=transformers_list)\n",
            "\n",
        ]
        # preprocess 정의 이후 pipe, param_grid, gs, fit, print 부분 유지
        idx = joined.find("pipe = Pipeline(")
        if idx != -1:
            rest = joined[idx:]
            new_source.append(rest)
        cell["source"] = new_source
        return nb
    return nb


def ensure_csv_path(nb, project_root):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        for i, line in enumerate(src):
            if isinstance(line, str) and "read_csv" in line and "coding_book_mapping" in line:
                base = os.path.join(project_root, "1. 초기 데이터 전처리", "3.coding_book_mapping.csv")
                src[i] = "origin = read_csv(r'" + base.replace("\\", "\\\\") + "', encoding='utf-8')\n"
                return nb
    return nb


def data_loading_no_drop(nb):
    """결측 제거 없이: origin2 -> df2 -> df3 -> x,y -> split."""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        joined = "".join(s if isinstance(s, str) else "" for s in src)
        if "origin2 = origin.drop" not in joined or "train_test_split" not in joined:
            continue
        new_source = [
            "origin2 = origin.drop(['dependent_wage_work'], axis=1)\n",
            "df2 = origin2.copy()\n",
            "df3 = df2.copy()\n",
            "\n",
            "yname = \"dependent_ecotype\"\n",
            "drop_for_leakage = [yname, 'work_ability_age']\n",
            "x = df3.drop(columns=[c for c in drop_for_leakage if c in df3.columns])\n",
            "y = df3[yname].astype(int)\n",
            "\n",
            "x_train, x_test, y_train, y_test = train_test_split(\n",
            "    x, y, test_size=0.25, random_state=52, stratify=y\n",
            ")\n",
            "x_train.shape, x_test.shape, y_train.shape, y_test.shape\n",
        ]
        cell["source"] = new_source
        return nb
    return nb


def main():
    os.makedirs(FOLDER_LOG_NO, exist_ok=True)
    os.makedirs(FOLDER_LOG_50, exist_ok=True)
    os.makedirs(FOLDER_LOG_80, exist_ok=True)
    for d in [FOLDER_LOG_NO, FOLDER_LOG_50, FOLDER_LOG_80]:
        os.makedirs(os.path.join(d, "results"), exist_ok=True)

    configs = [
        (FOLDER_LOG_NO, FOLDER_3, None, "로그변환 + 결측 제거 없음"),
        (FOLDER_LOG_50, FOLDER_50, None, "로그변환 + 결측 50% 초과 제거"),
        (FOLDER_LOG_80, FOLDER_80, None, "로그변환 + 결측 80% 초과 제거"),
    ]

    for dest_folder, source_folder, _, label in configs:
        print("\n---", label, "---")
        # 1) 로지스틱
        if dest_folder == FOLDER_LOG_NO:
            sgd_path = os.path.join(FOLDER_3, "new_경사하강법.ipynb")
            with open(sgd_path, "r", encoding="utf-8") as f:
                log_nb = json.load(f)
            log_nb = clear_outputs(log_nb)
            for cell in log_nb.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                src = cell.get("source", [])
                raw = "".join(s if isinstance(s, str) else "" for s in src)
                if "from analysis_utils import" in raw:
                    cell["source"] = add_sys_path_to_source(src, FOLDER_3)
                if "SGDClassifier" in raw:
                    full = raw.replace("from sklearn.linear_model import SGDClassifier", "from sklearn.linear_model import LogisticRegression")
                    full = full.replace("SGDClassifier(random_state=52, loss='log_loss', max_iter=2000)", "LogisticRegression(random_state=52, max_iter=1000)")
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
            for c in log_nb.get("cells", []):
                if c.get("cell_type") == "markdown" and c.get("source"):
                    txt = "".join(c["source"])
                    if "경사하강법" in txt and "독립" in txt:
                        c["source"] = ["# 로지스틱 회귀 분석 (독립 실행용)\n", "\n", label + " 버전.\n"]
                        break
            data_loading_no_drop(log_nb)
        else:
            src_logistic = os.path.join(source_folder, "new_로지스틱.ipynb")
            with open(src_logistic, "r", encoding="utf-8") as f:
                log_nb = json.load(f)
            log_nb = clear_outputs(log_nb)
            for c in log_nb.get("cells", []):
                if c.get("cell_type") == "markdown" and c.get("source"):
                    txt = "".join(c["source"])
                    if "로지스틱" in txt:
                        c["source"] = ["# 로지스틱 회귀 분석 (독립 실행용)\n", "\n", label + " 버전.\n"]
                        break
        add_function_transformer_import(log_nb)
        replace_pipeline_with_log_version(log_nb)
        ensure_csv_path(log_nb, PROJECT_ROOT)
        with open(os.path.join(dest_folder, "new_로지스틱.ipynb"), "w", encoding="utf-8") as f:
            json.dump(log_nb, f, ensure_ascii=False, indent=2)
        print("  wrote new_로지스틱.ipynb")

        # 2) 나머지 7개: source_folder에서 복사 후 로그 파이프라인 주입
        for name in MODEL_NOTEBOOKS:
            src_path = os.path.join(source_folder, name + ".ipynb")
            if not os.path.isfile(src_path):
                print("  skip (not found):", name)
                continue
            with open(src_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            nb = clear_outputs(nb)
            if dest_folder == FOLDER_LOG_NO:
                add_sys_path_to_notebook(nb, FOLDER_3)
                data_loading_no_drop(nb)
            add_function_transformer_import(nb)
            replace_pipeline_with_log_version(nb)
            ensure_csv_path(nb, PROJECT_ROOT)
            dest_path = os.path.join(dest_folder, name + ".ipynb")
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(nb, f, ensure_ascii=False, indent=2)
            print("  wrote", name + ".ipynb")

    # 종합.ipynb 3개
    for folder_path, pct_label in [
        (FOLDER_LOG_NO, "로그변환 + 결측 제거 없음"),
        (FOLDER_LOG_50, "로그변환 + 결측 50% 초과 제거"),
        (FOLDER_LOG_80, "로그변환 + 결측 80% 초과 제거"),
    ]:
        summary = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# 비선형 분류 모형 종합 비교 (" + pct_label + ")\n",
                        "\n",
                        "로지스틱, 경사하강법, KNN, SVM, 랜덤포레스트, XGBoost, LightGBM, CatBoost 결과를 불러와 **주요 성능 비교**, **과적합 여부**를 한눈에 봅니다.\n",
                        "\n",
                        "**사전 조건:** 각 `new_*.ipynb`를 실행해 `results/*.pkl`이 생성되어 있어야 합니다."
                    ]
                },
                {"cell_type": "markdown", "metadata": {}, "source": ["## 라이브러리 및 결과 로드"]},
                {
                    "cell_type": "code",
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import os\nimport pickle\nimport numpy as np\nimport pandas as pd\nfrom pandas import DataFrame\nimport matplotlib.pyplot as plt\nimport seaborn as sb\nimport shap\n\nmy_dpi = 100\n\nresults_dir = 'results'\npkl_names = ['new_로지스틱', 'new_경사하강법', 'new_KNN', 'new_SVM', 'new_랜덤포레스트', 'new_XGBoost', 'new_LightGBM', 'new_CatBoost']\n\nresults = {}\nfor name in pkl_names:\n    path = os.path.join(results_dir, name + '.pkl')\n    if os.path.isfile(path):\n        with open(path, 'rb') as f:\n            results[name] = pickle.load(f)\n        print(f'Loaded: {name}')\n    else:\n        print(f'Skip (not found): {path}')\n\nif not results:\n    raise SystemExit('결과 파일이 없습니다. 각 new_*.ipynb를 먼저 실행하세요.')"
                    ]
                },
                {"cell_type": "markdown", "metadata": {}, "source": ["## 1. 주요 성능 비교표"]},
                {
                    "cell_type": "code",
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "score_dfs = {k: v.get('score_df') for k, v in results.items() if v.get('score_df') is not None}\nif score_dfs:\n    summary_scores = pd.concat(score_dfs, axis=0)\n    display(summary_scores)\nelse:\n    print('score_df 없음')"
                    ]
                },
                {"cell_type": "markdown", "metadata": {}, "source": ["## 2. 성능 비교 막대 그래프 (AUC 기준)"]},
                {
                    "cell_type": "code",
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "if score_dfs:\n    names = list(results.keys())\n    aucs = []\n    for n in names:\n        df = results[n].get('score_df')\n        if df is not None and not df.empty:\n            a = df.get('AUC', df.get('roc_auc', pd.Series([None]))).iloc[0]\n            aucs.append(a)\n        else:\n            aucs.append(None)\n    fig, ax = plt.subplots(figsize=(10, 5), dpi=my_dpi)\n    ax.barh(names, aucs)\n    ax.set_xlabel('AUC')\n    ax.set_title('모델별 AUC (" + pct_label + ")')\n    plt.tight_layout()\n    plt.show()\nelse:\n    print('score_df 없음')"
                    ]
                },
                {"cell_type": "markdown", "metadata": {}, "source": ["## 3. 과적합 여부 요약"]},
                {
                    "cell_type": "code",
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "for name, data in results.items():\n    status = data.get('overfit_status', 'N/A')\n    print(f'{name}: {status}')"
                    ]
                },
            ],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11.0"}},
            "nbformat": 4,
            "nbformat_minor": 4,
        }
        with open(os.path.join(folder_path, "종합.ipynb"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("  wrote 종합.ipynb ->", folder_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
