# -*- coding: utf-8 -*-
"""
성별별 / 배우자 유무별 비교 노트북 생성.
6가지 데이터 버전(결측제거 없음/50%/80% × 로그없음/로그) × 전체 모델, subgroup AUC 비교.
"""
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FOLDERS = {
    "nodrop_nolog": os.path.join(PROJECT_ROOT, "3. 결측 변수 제거 없이 분석 진행"),
    "50_nolog": os.path.join(PROJECT_ROOT, "4. 결측 50% 초과 변수 제거 분석"),
    "80_nolog": os.path.join(PROJECT_ROOT, "5. 결측 80% 초과 변수 제거 분석"),
    "nodrop_log": os.path.join(PROJECT_ROOT, "6. 로그변환_결측제거없음"),
    "50_log": os.path.join(PROJECT_ROOT, "7. 로그변환_결측50초과제거"),
    "80_log": os.path.join(PROJECT_ROOT, "8. 로그변환_결측80초과제거"),
}
PKL_NAMES = [
    "new_로지스틱", "new_경사하강법", "new_KNN", "new_SVM",
    "new_랜덤포레스트", "new_XGBoost", "new_LightGBM", "new_CatBoost",
]
CSV_PATH = os.path.join(PROJECT_ROOT, "1. 초기 데이터 전처리", "3.coding_book_mapping.csv").replace("\\", "\\\\")


def make_subgroup_notebook(subgroup_type, subgroup_col, group_labels):
    """subgroup_type: '성별' or '배우자유무', subgroup_col: column name, group_labels: dict or list of (value, display_name)."""
    if subgroup_type == "성별":
        title = "성별별 비교"
        group_a, group_b = "남", "여"
        group_a_val, group_b_val = "남", "여"
        subgroup_comment = "w09gender1: 남/여"
    else:
        title = "배우자 유무별 비교"
        group_a, group_b = "배우자있음", "배우자없음"
        group_a_val, group_b_val = "혼인중", None  # 배우자있음=혼인중, 없음=그 외
        subgroup_comment = "w09marital: 혼인중=배우자있음, 그 외=배우자없음"

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# " + title + " (6가지 데이터 버전 × 전체 모델)\n",
                "\n",
                "**6가지 경우:**\n",
                "- 로그 없음: 결측 제거 없음 / 50% 초과 제거 / 80% 초과 제거\n",
                "- 로그 변환: 결측 제거 없음 / 50% 초과 제거 / 80% 초과 제거\n",
                "\n",
                "각 경우별로 저장된 모델(로지스틱, 경사하강법, KNN, SVM, 랜덤포레스트, XGBoost, LightGBM, CatBoost)을 불러와\n",
                "**전체·" + group_a + "·" + group_b + "** test AUC를 비교합니다.\n",
                "\n",
                "**사전 조건:** 3~8번 폴더에서 해당 버전의 `new_*.ipynb`를 실행해 `results/*.pkl`이 있어야 합니다."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 라이브러리 및 경로"]
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
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.metrics import roc_auc_score\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sb\n",
                "\n",
                "PROJECT_ROOT = r'" + PROJECT_ROOT.replace("\\", "\\\\") + "'\n",
                "CSV_PATH = os.path.join(PROJECT_ROOT, '1. 초기 데이터 전처리', '3.coding_book_mapping.csv')\n",
                "FOLDERS = {\n",
                "    'nodrop_nolog': os.path.join(PROJECT_ROOT, '3. 결측 변수 제거 없이 분석 진행'),\n",
                "    '50_nolog': os.path.join(PROJECT_ROOT, '4. 결측 50% 초과 변수 제거 분석'),\n",
                "    '80_nolog': os.path.join(PROJECT_ROOT, '5. 결측 80% 초과 변수 제거 분석'),\n",
                "    'nodrop_log': os.path.join(PROJECT_ROOT, '6. 로그변환_결측제거없음'),\n",
                "    '50_log': os.path.join(PROJECT_ROOT, '7. 로그변환_결측50초과제거'),\n",
                "    '80_log': os.path.join(PROJECT_ROOT, '8. 로그변환_결측80초과제거'),\n",
                "}\n",
                "PKL_NAMES = ['new_로지스틱', 'new_경사하강법', 'new_KNN', 'new_SVM', 'new_랜덤포레스트', 'new_XGBoost', 'new_LightGBM', 'new_CatBoost']\n",
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 데이터 로딩 (3가지 버전: 결측 제거 없음 / 50% / 80%)"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": [
                "from pandas import read_csv\n",
                "\n",
                "categorical_cols = [\n",
                "    'w09_fam1','w09_fam2','w09edu','w09gender1','w09marital','w09edu_s','w09ecoact_s','w09enu_type',\n",
                "    'w09ba069','w09bp1','w09c152','w09c001','w09c003','w09c005',\n",
                "    'w09chronic_a','w09chronic_b','w09chronic_c','w09chronic_d','w09chronic_e','w09chronic_f',\n",
                "    'w09chronic_g','w09chronic_h','w09chronic_i','w09chronic_j','w09chronic_k','w09chronic_l','w09chronic_m',\n",
                "    'w09c056','w09c068','w09c081','w09c082','w09c085','w09c102',\n",
                "    'w09smoke','w09alc','w09addic','w09c550',\n",
                "    'w09f001type','w09g031',\n",
                "    'w09cadd_19','w09c142','w09c143','w09c144','w09c145','w09c146','w09c147','w09c148','w09c149','w09c150','w09c151'\n",
                "]\n",
                "\n",
                "def load_and_split(threshold_pct=None):\n",
                "    origin = read_csv(CSV_PATH, encoding='utf-8')\n",
                "    origin_type_changed = origin.copy()\n",
                "    cat_for_type = [c for c in categorical_cols if c in origin_type_changed.columns]\n",
                "    origin_type_changed[cat_for_type] = origin_type_changed[cat_for_type].astype('category')\n",
                "    origin = origin_type_changed\n",
                "    origin2 = origin.drop(['dependent_wage_work'], axis=1)\n",
                "    yname = 'dependent_ecotype'\n",
                "    if threshold_pct is not None:\n",
                "        missing_rate = origin2.isnull().mean()\n",
                "        drop_high = [c for c in missing_rate[missing_rate > threshold_pct].index if c != yname]\n",
                "        origin2 = origin2.drop(columns=drop_high)\n",
                "    df3 = origin2.copy()\n",
                "    drop_for_leakage = [yname, 'work_ability_age']\n",
                "    x = df3.drop(columns=[c for c in drop_for_leakage if c in df3.columns])\n",
                "    y = df3[yname].astype(int)\n",
                "    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=52, stratify=y)\n",
                "    return x_train, x_test, y_train, y_test\n",
                "\n",
                "_, x_test_nodrop, _, y_test_nodrop = load_and_split(None)\n",
                "_, x_test_50, _, y_test_50 = load_and_split(0.5)\n",
                "_, x_test_80, _, y_test_80 = load_and_split(0.8)\n",
                "print('nodrop test:', x_test_nodrop.shape, '50% test:', x_test_50.shape, '80% test:', x_test_80.shape)\n",
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## " + subgroup_comment + " 기준 서브그룹 AUC 계산"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": [
                "def get_subgroup_auc(y_true, y_proba, subgroup_series, group_a_val, group_b_val):\n",
                "    res = {}\n",
                "    mask_a = (subgroup_series == group_a_val)\n",
                "    mask_b = (subgroup_series == group_b_val)\n",
                "    if mask_a.sum() > 0 and y_true[mask_a].nunique() > 1:\n",
                "        res['" + group_a + "'] = roc_auc_score(y_true[mask_a], y_proba[mask_a])\n",
                "    else:\n",
                "        res['" + group_a + "'] = np.nan\n",
                "    if mask_b.sum() > 0 and y_true[mask_b].nunique() > 1:\n",
                "        res['" + group_b + "'] = roc_auc_score(y_true[mask_b], y_proba[mask_b])\n",
                "    else:\n",
                "        res['" + group_b + "'] = np.nan\n",
                "    return res\n",
                "\n",
                "CASES = [\n",
                "    ('결측제거없음_로그없음', 'nodrop_nolog', x_test_nodrop, y_test_nodrop),\n",
                "    ('결측50%제거_로그없음', '50_nolog', x_test_50, y_test_50),\n",
                "    ('결측80%제거_로그없음', '80_nolog', x_test_80, y_test_80),\n",
                "    ('결측제거없음_로그변환', 'nodrop_log', x_test_nodrop, y_test_nodrop),\n",
                "    ('결측50%제거_로그변환', '50_log', x_test_50, y_test_50),\n",
                "    ('결측80%제거_로그변환', '80_log', x_test_80, y_test_80),\n",
                "]\n",
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": [
                "subgroup_col_name = '" + subgroup_col + "'\n",
                "group_a_val, group_b_val = " + repr((group_a_val, group_b_val)) + "  # " + subgroup_comment + "\n",
                "\n",
                "rows = []\n",
                "for case_label, folder_key, x_test, y_test in CASES:\n",
                "    folder = FOLDERS[folder_key]\n",
                "    if subgroup_col_name not in x_test.columns:\n",
                "        print(f'Skip {case_label}: no column {subgroup_col_name}')\n",
                "        continue\n",
                "    subgroup_series = x_test[subgroup_col_name]\n",
                "    if group_b_val is None:\n",
                "        subgroup_series_binary = subgroup_series == group_a_val\n",
                "        group_a_val_use, group_b_val_use = True, False\n",
                "    else:\n",
                "        group_a_val_use, group_b_val_use = group_a_val, group_b_val\n",
                "    for pkl_name in PKL_NAMES:\n",
                "        path = os.path.join(folder, 'results', pkl_name + '.pkl')\n",
                "        if not os.path.isfile(path):\n",
                "            continue\n",
                "        with open(path, 'rb') as f:\n",
                "            data = pickle.load(f)\n",
                "        est = data.get('estimator')\n",
                "        if est is None:\n",
                "            continue\n",
                "        try:\n",
                "            y_proba = est.predict_proba(x_test)[:, 1]\n",
                "        except Exception as e:\n",
                "            print(f'{case_label} {pkl_name}: predict_proba failed', e)\n",
                "            continue\n",
                "        overall_auc = roc_auc_score(y_test, y_proba)\n",
                "        if group_b_val is None:\n",
                "            mask_a = (subgroup_series == group_a_val)\n",
                "            mask_b = ~mask_a\n",
                "            auc_a = roc_auc_score(y_test[mask_a], y_proba[mask_a]) if mask_a.sum() > 0 and y_test[mask_a].nunique() > 1 else np.nan\n",
                "            auc_b = roc_auc_score(y_test[mask_b], y_proba[mask_b]) if mask_b.sum() > 0 and y_test[mask_b].nunique() > 1 else np.nan\n",
                "            row = {'case': case_label, 'model': pkl_name, 'AUC_전체': overall_auc, '" + group_a + "': auc_a, '" + group_b + "': auc_b}\n",
                "        else:\n",
                "            sg = get_subgroup_auc(y_test.values, y_proba, subgroup_series, group_a_val_use, group_b_val_use)\n",
                "            row = {'case': case_label, 'model': pkl_name, 'AUC_전체': overall_auc, '" + group_a + "': sg['" + group_a + "'], '" + group_b + "': sg['" + group_b + "']}\n",
                "        rows.append(row)\n",
                "\n",
                "df_sub = pd.DataFrame(rows)\n",
                "display(df_sub)\n",
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 요약: 경우별·모델별 AUC (전체 / " + group_a + " / " + group_b + ")"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": [
                "if len(df_sub) > 0:\n",
                "    pd.set_option('display.max_rows', None)\n",
                "    display(df_sub.round(4))\n",
                "    pivot_overall = df_sub.pivot_table(index='model', columns='case', values='AUC_전체')\n",
                "    display(pivot_overall.round(4))\n",
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 시각화: 경우별 전체 AUC (모델별)"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": [
                "if len(df_sub) > 0:\n",
                "    fig, ax = plt.subplots(figsize=(12, 6))\n",
                "    for model in df_sub['model'].unique():\n",
                "        d = df_sub[df_sub['model'] == model]\n",
                "        ax.plot(d['case'], d['AUC_전체'], 'o-', label=model)\n",
                "    ax.set_xticklabels(df_sub['case'].unique(), rotation=45, ha='right')\n",
                "    ax.set_ylabel('AUC')\n",
                "    ax.set_title('6가지 데이터 버전별 전체 Test AUC (모델별)')\n",
                "    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 시각화: " + group_a + " vs " + group_b + " AUC 비교 (모델·경우별)"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": [
                "if len(df_sub) > 0:\n",
                "    fig, ax = plt.subplots(figsize=(10, 5))\n",
                "    x = np.arange(len(df_sub))\n",
                "    w = 0.35\n",
                "    ax.bar(x - w/2, df_sub['" + group_a + "'], width=w, label='" + group_a + "')\n",
                "    ax.bar(x + w/2, df_sub['" + group_b + "'], width=w, label='" + group_b + "')\n",
                "    ax.set_xticks(x)\n",
                "    ax.set_xticklabels(df_sub['case'] + '_' + df_sub['model'], rotation=90, ha='right', fontsize=8)\n",
                "    ax.set_ylabel('AUC')\n",
                "    ax.set_title('" + group_a + " vs " + group_b + " Test AUC')\n",
                "    ax.legend()\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
            ]
        },
    ]

    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11.0"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main():
    os.makedirs(os.path.join(PROJECT_ROOT, "9. 성별별 비교"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "10. 배우자 유무별 비교"), exist_ok=True)

    nb_gender = make_subgroup_notebook("성별", "w09gender1", ("남", "여"))
    with open(os.path.join(PROJECT_ROOT, "9. 성별별 비교", "종합_성별별_6가지버전_비교.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb_gender, f, ensure_ascii=False, indent=2)
    print("Wrote 9. 성별별 비교/종합_성별별_6가지버전_비교.ipynb")

    nb_spouse = make_subgroup_notebook("배우자유무", "w09marital", ("배우자있음", "배우자없음"))
    # 배우자 유무는 w09marital에서 혼인중=있음, 그 외=없음으로 이진화
    # get_subgroup_auc 호출 부분을 배우자용으로 바꿔야 함. 현재 코드는 group_b_val=None일 때 혼인중 vs 그 외 처리함.
    with open(os.path.join(PROJECT_ROOT, "10. 배우자 유무별 비교", "종합_배우자유무별_6가지버전_비교.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb_spouse, f, ensure_ascii=False, indent=2)
    print("Wrote 10. 배우자 유무별 비교/종합_배우자유무별_6가지버전_비교.ipynb")
    print("Done.")


if __name__ == "__main__":
    main()
