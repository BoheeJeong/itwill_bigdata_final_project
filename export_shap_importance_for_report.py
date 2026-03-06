# -*- coding: utf-8 -*-
"""
3번 전처리 XGBoost SHAP 기반 변수 중요도(importance_ratio)를 계산하여
리포트 표 5용 수치를 출력합니다. 실행: python export_shap_importance_for_report.py
"""
import os
import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
PATH_3 = os.path.join(PROJECT_ROOT, "3. 결측 변수 제거 없이 분석 진행", "results", "new_XGBoost.pkl")


def compute_shap_summary(estimator, x_train, preprocess_step_name="preprocess", model_step_name="model"):
    import shap
    preprocess = estimator.named_steps[preprocess_step_name]
    inner = estimator.named_steps[model_step_name]
    X_tr = preprocess.transform(x_train)
    if hasattr(X_tr, "toarray"):
        X_tr = X_tr.toarray()
    feature_names = preprocess.get_feature_names_out()
    X_train_df = DataFrame(X_tr, columns=feature_names, index=x_train.index)

    explainer = shap.TreeExplainer(inner, data=X_train_df)
    shap_values = explainer.shap_values(X_train_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap_df = DataFrame(shap_values, columns=feature_names, index=x_train.index)
    total = shap_df.abs().mean().sum()
    summary_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": shap_df.abs().mean().values,
        "mean_shap": shap_df.mean().values,
    })
    summary_df["importance_ratio"] = summary_df["mean_abs_shap"] / total
    summary_df = summary_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return summary_df


def clean_feature_name(f):
    return f.replace("num__", "").replace("cat__", "")


def main():
    if not os.path.isfile(PATH_3):
        print("3번 new_XGBoost.pkl 없음. 먼저 '3. 결측 변수 제거 없이 분석 진행/new_XGBoost.ipynb'를 실행하세요.")
        return
    with open(PATH_3, "rb") as f:
        data_3 = pickle.load(f)
    estimator_3 = data_3.get("estimator")
    x_train_3 = data_3.get("x_train")
    if estimator_3 is None or x_train_3 is None:
        print("3번 pkl에 estimator 또는 x_train 없음.")
        return

    summary = compute_shap_summary(estimator_3, x_train_3)
    top_n = 20
    top = summary.head(top_n)

    print("=== 표 5용 SHAP 중요도(importance_ratio) 상위", top_n, "개 ===\n")
    print("순위\t변수명\t중요도(SHAP)")
    for i, row in top.iterrows():
        var = clean_feature_name(row["feature"])
        imp = round(float(row["importance_ratio"]), 3)
        print(f"{i+1}\t{var}\t{imp}")
    print("\n* 중요도 = mean_abs_shap / total (비율, 소수 셋째 자리)")


if __name__ == "__main__":
    main()
