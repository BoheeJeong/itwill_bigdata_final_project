# -*- coding: utf-8 -*-
"""
공통 분석 함수 모듈 (2. 로지스틱+성능평가+shap copy 노트북에서 추출)
다른 ipynb에서: from analysis_utils import hs_get_scores, my_shap_analysis, ... 로 사용
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from pandas import DataFrame, concat, merge
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    SGDRegressor,
    LogisticRegression,
)
import shap

# 노트북과 동일한 DPI (그래프 크기 제어)
my_dpi = 100


def hs_get_scores(estimator, x_test, y_true):
    if hasattr(estimator, "named_steps"):
        classname = estimator.named_steps["model"].__class__.__name__
    else:
        classname = estimator.__class__.__name__

    y_pred = estimator.predict(x_test)

    return DataFrame(
        {
            "결정계수(R2)": r2_score(y_true, y_pred),
            "평균절대오차(MAE)": mean_absolute_error(y_true, y_pred),
            "평균제곱오차(MSE)": mean_squared_error(y_true, y_pred),
            "평균오차(RMSE)": np.sqrt(mean_squared_error(y_true, y_pred)),
            "평균 절대 백분오차 비율(MAPE)": mean_absolute_percentage_error(
                y_true, y_pred
            ),
            "평균 비율 오차(MPE)": np.mean((y_true - y_pred) / y_true * 100),
        },
        index=[classname],
    )


def hs_describe(data, columns=None):
    num_columns = list(data.select_dtypes(include=np.number).columns)

    if not columns:
        columns = num_columns

    desc = data[columns].describe().T

    na_counts = data[columns].isnull().sum()
    desc.insert(1, "na_count", na_counts)
    desc.insert(2, "na_rate", (na_counts / len(data)) * 100)

    additional_stats = []

    for f in columns:
        if f not in num_columns:
            continue

        q1 = data[f].quantile(q=0.25)
        q3 = data[f].quantile(q=0.75)
        iqr = q3 - q1
        down = q1 - 1.5 * iqr
        up = q3 + 1.5 * iqr
        skew = data[f].skew()
        outlier_count = ((data[f] < down) | (data[f] > up)).sum()
        outlier_rate = (outlier_count / len(data)) * 100

        abs_skew = abs(skew)
        if abs_skew < 0.5:
            dist = "거의 대칭"
        elif abs_skew < 1.0:
            dist = "약한 우측 꼬리" if skew > 0 else "약한 좌측 꼬리"
        elif abs_skew < 2.0:
            dist = "중간 우측 꼬리" if skew > 0 else "중간 좌측 꼬리"
        else:
            dist = "극단 우측 꼬리" if skew > 0 else "극단 좌측 꼬리"

        if abs_skew < 0.5:
            log_need = "낮음"
        elif abs_skew < 1.0:
            log_need = "중간"
        else:
            log_need = "높음"

        additional_stats.append({
            "field": f,
            "iqr": iqr,
            "up": up,
            "down": down,
            "outlier_count": outlier_count,
            "outlier_rate": outlier_rate,
            "skew": skew,
            "dist": dist,
            "log_need": log_need,
        })

    additional_df = DataFrame(additional_stats).set_index("field")
    result = concat([desc, additional_df], axis=1)
    return result


def category_describe(data, columns=None):
    num_columns = data.select_dtypes(include=np.number).columns

    if not columns:
        columns = data.select_dtypes(
            include=["object", "category", "bool"]
        ).columns

    result = []
    summary = []

    for f in columns:
        if f in num_columns:
            continue

        value_counts = data[f].value_counts(dropna=False)

        for category, count in value_counts.items():
            rate = (count / len(data)) * 100
            result.append({
                "변수": f,
                "범주": category,
                "빈도": count,
                "비율(%)": round(rate, 2),
            })

        if len(value_counts) == 0:
            continue

        max_category = value_counts.index[0]
        max_count = value_counts.iloc[0]
        max_rate = (max_count / len(data)) * 100
        min_category = value_counts.index[-1]
        min_count = value_counts.iloc[-1]
        min_rate = (min_count / len(data)) * 100

        summary.append({
            "변수": f,
            "최대_범주": max_category,
            "최대_비율(%)": round(max_rate, 2),
            "최소_범주": min_category,
            "최소_비율(%)": round(min_rate, 2),
        })

    return DataFrame(result), DataFrame(summary).set_index("변수")


def hs_feature_importance(model, x, y):
    perm = permutation_importance(
        estimator=model,
        X=x,
        y=y,
        scoring="r2",
        n_repeats=30,
        random_state=42,
        n_jobs=-1,
    )

    perm_df = DataFrame(
        {
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        },
        index=x.columns,
    ).sort_values("importance_mean", ascending=False)

    df = perm_df.sort_values(by="importance_mean", ascending=False)
    figsize = (1280 / my_dpi, 600 / my_dpi)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=my_dpi)
    sb.barplot(data=df, x="importance_mean", y=df.index)
    ax.set_title("Permutation Importance")
    ax.set_xlabel("Permutation Importance (mean)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()

    return perm_df


def create_figure(figsize=(1280 / 100, 720 / 100), dpi=None):
    if dpi is None:
        dpi = my_dpi
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    return fig, ax


def finalize_plot(ax):
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()


def hs_learning_cv(
    estimator,
    x,
    y,
    scoring="neg_root_mean_squared_error",
    cv=5,
    train_sizes=None,
    n_jobs=-1,
):
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes, train_scores, cv_scores = learning_curve(
        estimator=estimator,
        X=x,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        shuffle=True,
        random_state=52,
    )

    if hasattr(estimator, "named_steps"):
        classname = estimator.named_steps["model"].__class__.__name__
    else:
        classname = estimator.__class__.__name__

    train_rmse = -train_scores
    cv_rmse = -cv_scores
    train_mean = train_rmse.mean(axis=1)
    cv_mean = cv_rmse.mean(axis=1)
    cv_std = cv_rmse.std(axis=1)

    final_train = train_mean[-1]
    final_cv = cv_mean[-1]
    final_std = cv_std[-1]
    gap_ratio = final_train / final_cv
    var_ratio = final_std / final_cv

    y_mean = y.mean()
    rmse_naive = np.sqrt(np.mean((y - y_mean) ** 2))
    std_y = y.std()
    min_r2 = 0.10
    rmse_r2 = np.sqrt((1 - min_r2) * np.var(y))
    some_threshold = min(rmse_naive, std_y, rmse_r2)

    if gap_ratio >= 0.95 and final_cv > some_threshold:
        status = "⚠️ 과소적합 (bias 큼)"
    elif gap_ratio <= 0.8:
        status = "⚠️ 과대적합 (variance 큼)"
    elif gap_ratio <= 0.95 and var_ratio <= 0.10:
        status = "✅ 일반화 양호"
    elif var_ratio > 0.15:
        status = "⚠️ 데이터 부족 / 분산 큼"
    else:
        status = "⚠️ 판단 유보"

    result_df = DataFrame(
        {
            "Train RMSE": [final_train],
            "CV RMSE 평균": [final_cv],
            "CV RMSE 표준편차": [final_std],
            "Train/CV 비율": [gap_ratio],
            "CV 변동성 비율": [var_ratio],
            "판정 결과": [status],
        },
        index=[classname],
    )

    fig, ax = create_figure()
    sb.lineplot(
        x=train_sizes,
        y=train_mean,
        marker="o",
        markeredgecolor="#ffffff",
        label="Train RMSE",
    )
    sb.lineplot(
        x=train_sizes,
        y=cv_mean,
        marker="o",
        markeredgecolor="#ffffff",
        label="Train RMSE",
    )
    ax.set_xlabel("RMSE", fontsize=8, labelpad=5)
    ax.set_ylabel("학습곡선 (Learning Curve)", fontsize=8, labelpad=5)
    ax.grid(True, alpha=0.3)
    finalize_plot(ax)

    return result_df


def hs_get_score_cv(
    estimator,
    x_test,
    y_test,
    x_origin,
    y_origin,
    scoring="neg_root_mean_squared_error",
    cv=5,
    train_sizes=None,
    n_jobs=-1,
):
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    score_df = hs_get_scores(estimator, x_test, y_test)
    cv_df = hs_learning_cv(
        estimator,
        x_origin,
        y_origin,
        scoring=scoring,
        cv=cv,
        train_sizes=train_sizes,
        n_jobs=n_jobs,
    )
    return merge(score_df, cv_df, left_index=True, right_index=True)


def my_shap_analysis(
    model,
    x: DataFrame,
    plot: bool = True,
    width: int = 1600,
):
    if isinstance(model, Pipeline):
        estimator = model.named_steps["model"]
        is_pipeline = True
    else:
        estimator = model
        is_pipeline = False

    is_linear_model = isinstance(
        estimator,
        (LinearRegression, Ridge, Lasso, SGDRegressor, LogisticRegression),
    )

    x_df = x.copy()
    columns = x.columns.tolist()
    indexs = x.index.tolist()

    if is_linear_model:
        if is_pipeline:
            for name, step in list(model.named_steps.items()):
                if name == "model":
                    continue
                x_df = step.transform(x_df)
        masker = shap.maskers.Independent(x_df)
        explainer = shap.LinearExplainer(estimator, masker=masker)
    else:
        explainer = shap.TreeExplainer(estimator)

    shap_values = explainer.shap_values(x_df)

    shap_df = DataFrame(
        shap_values,
        columns=columns,
        index=indexs,
    )

    summary_df = DataFrame(
        {
            "feature": shap_df.columns,
            "mean_abs_shap": shap_df.abs().mean().values,
            "mean_shap": shap_df.mean().values,
            "std_shap": shap_df.std().values,
        }
    )

    summary_df["direction"] = np.where(
        summary_df["mean_shap"] > 0,
        "양(+) 경향",
        np.where(summary_df["mean_shap"] < 0, "음(-) 경향", "혼합/미약"),
    )
    summary_df["cv"] = summary_df["std_shap"] / (
        summary_df["mean_abs_shap"] + 1e-9
    )
    summary_df["variability"] = np.where(
        summary_df["cv"] < 1, "stable", "variable"
    )
    summary_df = (
        summary_df.sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    total_importance = summary_df["mean_abs_shap"].sum()
    summary_df["importance_ratio"] = summary_df["mean_abs_shap"] / total_importance
    summary_df["importance_cumsum"] = summary_df["importance_ratio"].cumsum()
    summary_df["is_important"] = np.where(
        summary_df["importance_cumsum"] <= 0.80, "core", "secondary"
    )

    if plot:
        shap.summary_plot(shap_values, x_df, show=False)
        fig = plt.gcf()
        fig.set_size_inches(width / my_dpi, len(summary_df) * 70 / my_dpi)
        ax = fig.get_axes()[0]
        ax.set_title("SHAP Summary Plot", fontsize=10, pad=10)
        plt.xlabel("SHAP value", fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.close()

    return summary_df, shap_values


def hs_shap_dependence_analysis(
    summary_df: DataFrame,
    shap_values,
    x_train: DataFrame,
    include_secondary: bool = False,
    width: int = 1600,
    height: int = 800,
):
    main_features = summary_df[
        (summary_df["is_important"] == "core")
        & (summary_df["variability"] == "variable")
    ]["feature"].tolist()

    interaction_features = summary_df[
        summary_df["is_important"] == "core"
    ]["feature"].tolist()

    if include_secondary and len(interaction_features) < 2:
        interaction_features.extend(
            summary_df[summary_df["is_important"] == "secondary"]["feature"].tolist()
        )

    pairs = []
    for f in main_features:
        for inter in interaction_features:
            if f != inter:
                pairs.append((f, inter))

    importance_rank = {}
    for i, row in summary_df.iterrows():
        importance_rank[row["feature"]] = i
    pairs = sorted(pairs, key=lambda x: importance_rank.get(x[0], 999))

    for feature_name, interaction_name in pairs:
        shap.dependence_plot(
            feature_name,
            shap_values,
            x_train,
            interaction_index=interaction_name,
            show=False,
        )
        fig = plt.gcf()
        fig.set_size_inches(width / my_dpi, height / my_dpi)
        plt.title(
            f"SHAP Dependence Plot: {feature_name} × {interaction_name}",
            fontsize=10,
            pad=10,
        )
        plt.xlabel(feature_name, fontsize=10)
        plt.ylabel(f"SHAP value for {feature_name}", fontsize=10)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.close()

    return pairs
