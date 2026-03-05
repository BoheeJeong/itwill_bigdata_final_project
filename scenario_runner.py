import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)


PROJECT_ROOT = r"c:\itwill_bigdata_final_project-main\itwill_bigdata_final_project"
CSV_PATH = os.path.join(
    PROJECT_ROOT, "1. 초기 데이터 전처리", "3.coding_book_mapping.csv"
)

# 3번 분석과 동일한 범주의 명목형 컬럼 목록
categorical_cols = [
    "w09_fam1",
    "w09_fam2",
    "w09edu",
    "w09gender1",
    "w09marital",
    "w09edu_s",
    "w09ecoact_s",
    "w09enu_type",
    "w09ba069",
    "w09bp1",
    "w09c152",
    "w09c001",
    "w09c003",
    "w09c005",
    "w09chronic_a",
    "w09chronic_b",
    "w09chronic_c",
    "w09chronic_d",
    "w09chronic_e",
    "w09chronic_f",
    "w09chronic_g",
    "w09chronic_h",
    "w09chronic_i",
    "w09chronic_j",
    "w09chronic_k",
    "w09chronic_l",
    "w09chronic_m",
    "w09c056",
    "w09c068",
    "w09c081",
    "w09c082",
    "w09c085",
    "w09c102",
    "w09smoke",
    "w09alc",
    "w09addic",
    "w09c550",
    "w09f001type",
    "w09g031",
    "w09cadd_19",
    "w09c142",
    "w09c143",
    "w09c144",
    "w09c145",
    "w09c146",
    "w09c147",
    "w09c148",
    "w09c149",
    "w09c150",
    "w09c151",
]


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list, list]:
    """
    3번(결측 제거 없음, 로그 없음)과 동일한 방식으로 데이터 로딩 및 train/test 분할.
    """
    origin = pd.read_csv(CSV_PATH, encoding="utf-8")
    origin_type_changed = origin.copy()
    cat_for_type = [
        c for c in categorical_cols if c in origin_type_changed.columns
    ]
    origin_type_changed[cat_for_type] = origin_type_changed[cat_for_type].astype(
        "category"
    )
    origin = origin_type_changed

    origin2 = origin.drop(["dependent_wage_work"], axis=1)
    yname = "dependent_ecotype"
    # 경제활동 여부와 직접 연결될 수 있는 work_ability_age는 누수 방지를 위해 제외
    drop_for_leakage = [yname, "work_ability_age"]
    x = origin2.drop(columns=[c for c in drop_for_leakage if c in origin2.columns])
    y = origin2[yname].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=52, stratify=y
    )
    cat_cols = x_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = x_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return x_train, x_test, y_train, y_test, cat_cols, num_cols


class MedianImputerWithMissingIndicator(BaseEstimator, TransformerMixin):
    """
    연속형 결측을 중앙값으로 채우고, 결측 여부(0/1) 컬럼을 추가한 뒤 스케일링.
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.imputer.fit(X)
        X_imputed = self.imputer.transform(X)
        self.scaler.fit(X_imputed)
        return self

    def transform(self, X):
        X_imputed = self.imputer.transform(X)
        missing = np.isnan(X).astype(np.float64)
        scaled = self.scaler.transform(X_imputed)
        return np.hstack([scaled, missing])


def build_numeric_pipe(scenario: str) -> Pipeline:
    """
    시나리오별 연속형 전처리 파이프라인 생성.

    scenario:
        - \"median\"       : 중앙값 대체 + 스케일
        - \"mean\"         : 평균 대체 + 스케일
        - \"knn\"          : KNNImputer + 스케일
        - \"iterative\"    : IterativeImputer + 스케일
        - \"median_flag\"  : median + 결측 여부 플래그 추가
    """
    scenario = scenario.lower()

    if scenario == "median":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    if scenario == "mean":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )
    if scenario == "knn":
        return Pipeline(
            [
                ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
                ("scaler", StandardScaler()),
            ]
        )
    if scenario == "iterative":
        return Pipeline(
            [
                ("imputer", IterativeImputer(max_iter=10, random_state=52)),
                ("scaler", StandardScaler()),
            ]
        )
    if scenario == "median_flag":
        return Pipeline(
            [
                ("imputer_with_flag", MedianImputerWithMissingIndicator()),
            ]
        )

    raise ValueError(f"Unknown scenario: {scenario}")


def run_model(
    scenario: str,
    estimator: BaseEstimator,
    save_dir: str,
    save_name: str,
    verbose: bool = True,
):
    """
    공통 데이터 로딩/전처리/학습/평가/저장을 한 번에 수행.

    Parameters
    ----------
    scenario : str
        연속형 결측 처리 시나리오 이름 (median, mean, knn, iterative, median_flag)
    estimator : BaseEstimator
        학습에 사용할 sklearn 모델 인스턴스 (로지스틱, SVM, 랜덤포레스트 등)
    save_dir : str
        pkl을 저장할 디렉토리(절대경로 추천)
    save_name : str
        저장 파일명(확장자 .pkl 제외). 예: \"new_로지스틱\"
    verbose : bool
        True이면 주요 결과를 print
    """
    x_train, x_test, y_train, y_test, cat_cols, num_cols = load_data()

    categorical_pipe = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="Missing"),
            ),
            (
                "onehot",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
            ),
        ]
    )

    numeric_pipe = build_numeric_pipe(scenario)

    preprocess = ColumnTransformer(
        [
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )

    pipe = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", estimator),
        ]
    )

    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)

    # 확률 예측이 가능한 모델이면 AUC, log_loss까지 계산
    y_proba = None
    auc = None
    ll = None
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        y_proba = pipe.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        ll = log_loss(y_test, y_proba)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "scenario": scenario,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "log_loss": ll,
    }

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{save_name}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "pipeline": pipe,
                "metrics": metrics,
                "scenario": scenario,
            },
            f,
        )

    if verbose:
        print(f"[{scenario}] Saved result to: {save_path}")
        print("metrics:", metrics)

    return metrics

