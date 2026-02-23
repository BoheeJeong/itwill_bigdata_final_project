# -*- coding: utf-8 -*-
"""
비선형 분류 모형별 독립 노트북 생성 (new_*.ipynb)
각 노트북: 데이터 로딩, 전처리, 해당 모형 파이프라인, GridSearchCV, 성능평가, 과적합 확인, SHAP, 결과 저장
"""
import os
import json

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = r"C:\\itwill_bigdata_final_project-main\\itwill_bigdata_final_project\\1. 초기 데이터 전처리\\3.coding_book_mapping.csv"
CAT_COLS_STR = """categorical_cols = [
    'w09_fam1','w09_fam2','w09edu','w09gender1','w09marital','w09edu_s','w09ecoact_s','w09enu_type',
    'w09ba069','w09bp1','w09c152','w09c001','w09c003','w09c005',
    'w09chronic_a','w09chronic_b','w09chronic_c','w09chronic_d','w09chronic_e','w09chronic_f',
    'w09chronic_g','w09chronic_h','w09chronic_i','w09chronic_j','w09chronic_k','w09chronic_l','w09chronic_m',
    'w09c056','w09c068','w09c081','w09c082','w09c085','w09c102',
    'w09smoke','w09alc','w09addic','w09c550',
    'w09f001type','w09g031',
    'w09cadd_19','w09c142','w09c143','w09c144','w09c145','w09c146','w09c147','w09c148','w09c149','w09c150','w09c151'
]"""


def ce(s):
    return [line + "\n" for line in s.strip().split("\n")]


def make_cell_md(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": ce(lines) if isinstance(lines, str) else lines}


def make_cell_code(lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ce(lines) if isinstance(lines, str) else lines}


# 공통 셀 (데이터 로딩 ~ 컬럼 분리)
COMMON_HEAD = [
    ("# {title}\n\n독립 실행용. 데이터 로딩부터 성능평가·과적합 확인·SHAP·결과 저장까지 수행."),
    ("## 라이브러리"),
    ("""import pandas as pd
import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    log_loss,
)
import shap
import os

my_dpi = 100"""),
    ("""# 공통 함수 (analysis_utils)
from analysis_utils import (
    create_figure, finalize_plot, my_dpi,
)"""),
    ("## 데이터 로딩"),
    ("""from pandas import read_csv
origin = read_csv(r'{}', encoding='utf-8')
origin.head()""".format(DATA_PATH)),
    (CAT_COLS_STR),
    ("""origin_type_changed = origin.copy()
cat_cols_for_type = [c for c in categorical_cols if c in origin_type_changed.columns]
origin_type_changed[cat_cols_for_type] = origin_type_changed[cat_cols_for_type].astype("category")
origin = origin_type_changed.copy()"""),
    ("""origin2 = origin.drop(['dependent_wage_work'], axis=1)
df2 = origin2.copy()
df3 = df2.copy()
yname = "dependent_ecotype"
x = df3.drop(columns=[yname])
y = df3[yname].astype(int)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=52, stratify=y
)
x_train.shape, x_test.shape, y_train.shape, y_test.shape"""),
    ("## #01 컬럼 타입 분리"),
    ("""cat_cols = x_train.select_dtypes(include=["object", "category"]).columns
num_cols = x_train.select_dtypes(include=["int64", "float64"]).columns
print("categorical:", len(cat_cols))
print("numeric:", len(num_cols))"""),
]

# 모형별: (#02 파이프라인+GridSearch) 내용
MODELS = {
    "new_경사하강법": {
        "title": "경사하강법 (SGDClassifier)",
        "import_add": "from sklearn.linear_model import SGDClassifier",
        "pipe_model": "(\"model\", SGDClassifier(random_state=52, loss='log_loss', max_iter=2000))",
        "param_grid": """param_grid = {
    "model__alpha": [0.0001, 0.001, 0.01],
    "model__max_iter": [1000, 2000],
    "model__penalty": ["l2", "l1"],
    "model__class_weight": [None, "balanced"],
}""",
        "shap_type": "linear",  # LinearExplainer
    },
    "new_KNN": {
        "title": "KNN (KNeighborsClassifier)",
        "import_add": "from sklearn.neighbors import KNeighborsClassifier",
        "pipe_model": "(\"model\", KNeighborsClassifier())",
        "param_grid": """param_grid = {
    "model__n_neighbors": [5, 10, 20, 30, 50],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],
}""",
        "shap_type": "kernel",
    },
    "new_SVM": {
        "title": "SVM (SVC)",
        "import_add": "from sklearn.svm import SVC",
        "pipe_model": "(\"model\", SVC(probability=True, random_state=52))",
        "param_grid": """param_grid = {
    "model__C": [0.1, 1, 10],
    "model__kernel": ["rbf", "poly"],
    "model__gamma": ["scale", "auto"],
    "model__class_weight": [None, "balanced"],
}""",
        "shap_type": "kernel",
    },
    "new_XGBoost": {
        "title": "XGBoost (XGBClassifier)",
        "import_add": "from xgboost import XGBClassifier",
        "pipe_model": "(\"model\", XGBClassifier(random_state=52, use_label_encoder=False, eval_metric='logloss'))",
        "param_grid": """param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
    "model__scale_pos_weight": [1, 2],
}""",
        "shap_type": "tree",
    },
    "new_LightGBM": {
        "title": "LightGBM (LGBMClassifier)",
        "import_add": "import lightgbm as lgb\nfrom lightgbm import LGBMClassifier",
        "pipe_model": "(\"model\", LGBMClassifier(random_state=52, verbose=-1))",
        "param_grid": """param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.05, 0.1],
    "model__num_leaves": [31, 63],
    "model__class_weight": [None, "balanced"],
}""",
        "shap_type": "tree",
    },
    "new_CatBoost": {
        "title": "CatBoost (CatBoostClassifier)",
        "import_add": "from catboost import CatBoostClassifier",
        "pipe_model": "(\"model\", CatBoostClassifier(random_state=52, verbose=0))",
        "param_grid": """param_grid = {
    "model__iterations": [100, 200],
    "model__depth": [3, 5, 7],
    "model__learning_rate": [0.05, 0.1],
    "model__l2_leaf_reg": [1, 3],
}""",
        "shap_type": "tree",
    },
}


def pipeline_section(key):
    m = MODELS[key]
    lib = m["import_add"]
    return (
        "## #02 전처리 + " + m["title"] + " 파이프라인 & GridSearchCV",
        """numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
])
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ]
)
pipe = Pipeline([
    ("preprocess", preprocess),
    """ + m["pipe_model"] + """
])
""" + m["param_grid"] + """
gs = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
gs.fit(x_train, y_train)
estimator = gs.best_estimator_
print("Best CV AUC:", gs.best_score_)
print("Best params:", gs.best_params_)"""
    )


def shap_cell(key):
    m = MODELS[key]
    t = m["shap_type"]
    if t == "tree":
        return """X_train_transformed = estimator.named_steps["preprocess"].transform(x_train)
feature_names = estimator.named_steps["preprocess"].get_feature_names_out()
inner_model = estimator.named_steps["model"]
X_train_df = DataFrame(X_train_transformed, columns=feature_names, index=x_train.index)
plot_df = X_train_df
explainer = shap.TreeExplainer(inner_model, data=X_train_df)
shap_values = explainer.shap_values(X_train_df)
if isinstance(shap_values, list):
    shap_values = shap_values[1]
shap_df = DataFrame(shap_values, columns=feature_names, index=x_train.index)
summary_df = DataFrame({
    "feature": shap_df.columns,
    "mean_abs_shap": shap_df.abs().mean().values,
    "mean_shap": shap_df.mean().values,
    "std_shap": shap_df.std().values,
})
summary_df["direction"] = np.where(summary_df["mean_shap"] > 0, "양(+) 경향",
    np.where(summary_df["mean_shap"] < 0, "음(-) 경향", "혼합/미약"))
summary_df = summary_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
total_importance = summary_df["mean_abs_shap"].sum()
summary_df["importance_ratio"] = summary_df["mean_abs_shap"] / total_importance
summary_df["importance_cumsum"] = summary_df["importance_ratio"].cumsum()
summary_df["is_important"] = np.where(summary_df["importance_cumsum"] <= 0.80, "core", "secondary")
display(summary_df.head(20))"""
    if t == "linear":
        return """X_train_transformed = estimator.named_steps["preprocess"].transform(x_train)
feature_names = estimator.named_steps["preprocess"].get_feature_names_out()
X_train_df = DataFrame(X_train_transformed, columns=feature_names, index=x_train.index)
plot_df = X_train_df
inner_model = estimator.named_steps["model"]
masker = shap.maskers.Independent(X_train_df)
explainer = shap.LinearExplainer(inner_model, masker=masker)
shap_values = explainer.shap_values(X_train_df)
shap_df = DataFrame(shap_values, columns=feature_names, index=x_train.index)
summary_df = DataFrame({
    "feature": shap_df.columns,
    "mean_abs_shap": shap_df.abs().mean().values,
    "mean_shap": shap_df.mean().values,
    "std_shap": shap_df.std().values,
})
summary_df["direction"] = np.where(summary_df["mean_shap"] > 0, "양(+) 경향",
    np.where(summary_df["mean_shap"] < 0, "음(-) 경향", "혼합/미약"))
summary_df = summary_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
display(summary_df.head(20))"""
    # kernel: use a sample for speed
    return """X_train_transformed = estimator.named_steps["preprocess"].transform(x_train)
feature_names = estimator.named_steps["preprocess"].get_feature_names_out()
X_train_df = DataFrame(X_train_transformed, columns=feature_names, index=x_train.index)
sample_size = min(200, len(X_train_df))
X_sample = X_train_df.sample(n=sample_size, random_state=52)
background = shap.sample(X_train_df, 100)
explainer = shap.KernelExplainer(estimator.predict_proba, background)
shap_values = explainer.shap_values(X_sample, nsamples=50)
if isinstance(shap_values, list):
    shap_values = shap_values[1]
shap_df = DataFrame(shap_values, columns=feature_names, index=X_sample.index)
summary_df = DataFrame({
    "feature": shap_df.columns,
    "mean_abs_shap": shap_df.abs().mean().values,
    "mean_shap": shap_df.mean().values,
    "std_shap": shap_df.std().values,
})
summary_df = summary_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
plot_df = X_sample
display(summary_df.head(20))"""


def build_notebook(key):
    m = MODELS[key]
    title = m["title"]
    cells = []

    # head: title
    cells.append(make_cell_md(COMMON_HEAD[0][0].format(title=title)))
    cells.append(make_cell_md(COMMON_HEAD[1][0]))
    # lib: base + model-specific
    cells.append(make_cell_code(COMMON_HEAD[2][0] + "\n" + m["import_add"]))
    cells.append(make_cell_code(COMMON_HEAD[3][0]))
    md_indices = {4, 9}  # "## 데이터 로딩", "## #01 컬럼 타입 분리"
    for i in range(4, len(COMMON_HEAD)):
        cells.append(make_cell_md(COMMON_HEAD[i][0]) if i in md_indices else make_cell_code(COMMON_HEAD[i][0]))

    # #02 pipeline
    sec_md, sec_code = pipeline_section(key)
    cells.append(make_cell_md(sec_md))
    cells.append(make_cell_code(sec_code))

    # #03 예측
    cells.append(make_cell_md("## #03 예측값"))
    cells.append(make_cell_code("""y_pred = estimator.predict(x_test)
y_pred_proba = estimator.predict_proba(x_test)
y_pred_proba_1 = y_pred_proba[:, 1]
y_pred[:5], y_pred_proba_1[:5]"""))

    # #04 성능평가
    cells.append(make_cell_md("## #04 성능 평가"))
    cells.append(make_cell_code("""cm = confusion_matrix(y_test, y_pred)
((TN, FP), (FN, TP)) = cm
cmdf = DataFrame(cm, index=['Actual 0 (TN/FP)', 'Actual 1 (FN/TP)'], columns=['Predicted (Negative)', 'Predicted (Positive)'])
display(cmdf)"""))
    cells.append(make_cell_code("""width_px, height_px = 800, 600
fig, ax = plt.subplots(1, 1, figsize=(width_px / my_dpi, height_px / my_dpi), dpi=my_dpi)
sb.heatmap(data=cmdf, annot=True, fmt="d", linewidth=0.5, cmap="PuOr")
ax.set_xlabel("")
ax.set_ylabel("")
ax.xaxis.tick_top()
plt.tight_layout()
plt.show()
plt.close()"""))
    cells.append(make_cell_code("""accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
tpr = recall_score(y_test, y_pred)
fpr = FP / (TN + FP)
tnr = 1 - fpr
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba_1)
y_null = np.ones_like(y_test) * y_test.mean()
log_loss_test = -log_loss(y_test, y_pred_proba, normalize=False)
log_loss_null = -log_loss(y_test, y_null, normalize=False)
pseudo_r2 = 1 - (log_loss_test / log_loss_null)
print("Accuracy:", accuracy, "Precision:", precision, "Recall:", tpr, "FPR:", fpr, "TNR:", tnr, "F1:", f1, "AUC:", auc, "Pseudo R2:", pseudo_r2)"""))
    cells.append(make_cell_md("### ROC 곡선"))
    cells.append(make_cell_code("""roc_fpr, roc_tpr, _ = roc_curve(y_test, y_pred_proba_1)
fig, ax = plt.subplots(1, 1, figsize=(1000 / my_dpi, 900 / my_dpi), dpi=my_dpi)
sb.lineplot(x=roc_fpr, y=roc_tpr)
sb.lineplot(x=[0, 1], y=[0, 1], color='red', linestyle=":", alpha=0.5)
plt.fill_between(x=roc_fpr, y1=roc_tpr, alpha=0.1)
ax.grid(True, alpha=0.3)
ax.set_title(f"AUC={auc:.4f}", fontsize=10, pad=4)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()
plt.close()"""))
    cells.append(make_cell_md("### 결과표"))
    cells.append(make_cell_code("""if hasattr(estimator, "named_steps"):
    classname = estimator.named_steps["model"].__class__.__name__
else:
    classname = estimator.__class__.__name__
score_df = DataFrame({
    "의사결정계수(R2)": [round(pseudo_r2, 3)],
    "정확도(Accuracy)": [round(accuracy, 3)],
    "정밀도(Precision)": [round(precision, 3)],
    "재현율(Recall)": [round(tpr, 3)],
    "위양성율(Fallout)": [round(fpr, 3)],
    "특이성(TNR)": [round(tnr, 3)],
    "F1 Score": [round(f1, 3)],
    "AUC": [round(auc, 3)],
}, index=[classname])
score_df"""))

    # #05 Learning curve
    cells.append(make_cell_md("## #05 Learning Curve & 과적합 판정"))
    cells.append(make_cell_code("""train_sizes = np.linspace(0.1, 1.0, 10)
sizes, train_scores, cv_scores = learning_curve(
    estimator=estimator, X=x_train, y=y_train.astype(int),
    train_sizes=train_sizes, cv=5, scoring="roc_auc", n_jobs=-1, shuffle=True, random_state=52
)
train_mean = train_scores.mean(axis=1)
cv_mean = cv_scores.mean(axis=1)
cv_std = cv_scores.std(axis=1)
final_train = train_mean[-1]
final_cv = cv_mean[-1]
final_std = cv_std[-1]
gap_ratio = final_train - final_cv
print("Final Train AUC:", final_train, "Final CV AUC:", final_cv, "Gap(Train-CV):", gap_ratio)"""))
    cells.append(make_cell_code("""if final_train < 0.6 and final_cv < 0.6:
    status = "⚠ 과소적합"
elif gap_ratio > 0.1:
    status = "⚠ 과대적합"
elif gap_ratio <= 0.05 and final_std <= 0.05:
    status = "✅ 일반화 양호"
elif final_std > 0.1:
    status = "⚠ 데이터 부족"
else:
    status = "⚠ 판단 보류"
result_df = DataFrame({
    "Train ROC_AUC 평균": [round(final_train, 3)],
    "CV ROC_AUC 평균": [round(final_cv, 3)],
    "CV ROC_AUC 표준편차": [round(final_std, 3)],
    "Train/CV 갭": [round(gap_ratio, 3)],
    "판정 결과": [status],
}, index=[classname])
result_df"""))
    cells.append(make_cell_code("""fig, ax = plt.subplots(1, 1, figsize=(1600 / my_dpi, 960 / my_dpi), dpi=my_dpi)
sb.lineplot(x=train_sizes, y=train_mean, marker="o", markeredgecolor="#ffffff", label="Train ROC_AUC")
sb.lineplot(x=train_sizes, y=cv_mean, marker="o", markeredgecolor="#ffffff", label="CV ROC_AUC")
ax.fill_between(train_sizes, train_mean - train_scores.std(axis=1), train_mean + train_scores.std(axis=1), alpha=0.1)
ax.fill_between(train_sizes, cv_mean - cv_std, cv_mean + cv_std, alpha=0.1)
ax.set_xlabel("Train size")
ax.set_ylabel("ROC_AUC")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()"""))

    # #06 SHAP
    cells.append(make_cell_md("## #06 SHAP\n\n" + title + " Explainer 적용."))
    cells.append(make_cell_code(shap_cell(key)))
    cells.append(make_cell_code("""shap.summary_plot(shap_values, plot_df, show=False)
fig = plt.gcf()
fig.set_size_inches(16, 8)
plt.title("SHAP Summary Plot (" + title + ")", fontsize=10, pad=10)
plt.tight_layout()
plt.show()
plt.close()"""))

    # 결과 저장
    cells.append(make_cell_md("## 결과 저장 (종합.ipynb에서 사용)"))
    cells.append(make_cell_code("""os.makedirs('results', exist_ok=True)
import pickle
save_name = '" + key + "'
with open(os.path.join('results', save_name + '.pkl'), 'wb') as f:
    pickle.dump({
        'model_name': classname,
        'score_df': score_df,
        'result_df': result_df,
        'overfit_status': status,
        'estimator': estimator,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'auc': auc,
    }, f)
print('Saved results to results/' + save_name + '.pkl')"""))

    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11.0"}}, "nbformat": 4, "nbformat_minor": 4}


def main():
    for key in MODELS:
        nb = build_notebook(key)
        path = os.path.join(BASE, key + ".ipynb")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=2)
        print("Wrote", path)
    print("Done.")


if __name__ == "__main__":
    main()
