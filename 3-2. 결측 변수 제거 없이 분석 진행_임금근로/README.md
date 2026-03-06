# 3-2. 결측 변수 제거 없이 분석 진행 (임금근로)

- **3번 폴더와 동일한 전처리·데이터 분할**을 유지하고, **종속변수만** `dependent_wage_work`(임금근로 여부)로 변경한 분석입니다.
- 3번: 종속변수 `dependent_ecotype` (경제활동 여부)
- 3-2: 종속변수 `dependent_wage_work` (임금근로 여부)

## 실행 방법

1. 이 폴더(`3-2. 결측 변수 제거 없이 분석 진행_임금근로`)를 작업 디렉터리로 두고, 각 `new_*.ipynb`를 순서대로 실행합니다.
2. 결과는 **`results/`** 폴더에 `new_로지스틱.pkl`, `new_경사하강법.pkl`, … 형태로 저장됩니다.

## 노트북 목록

- new_로지스틱.ipynb
- new_경사하강법.ipynb
- new_KNN.ipynb
- new_SVM.ipynb
- new_랜덤포레스트.ipynb
- new_XGBoost.ipynb
- new_LightGBM.ipynb
- new_CatBoost.ipynb

## 데이터

- CSV 경로는 3번과 동일하게 프로젝트 루트 기준 `1. 초기 데이터 전처리/3.coding_book_mapping.csv` 를 사용합니다. (노트북 내 절대 경로 또는 상위 폴더 기준 경로로 로드)
