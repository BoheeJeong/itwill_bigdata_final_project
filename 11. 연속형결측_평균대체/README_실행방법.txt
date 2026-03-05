11~14번 폴더 노트북 일괄 실행 방법
=====================================

(1) 프로젝트 루트에서 스크립트 실행 (권장)
   - 명령: python run_11_14_notebooks.py
   - 11~14번 각 폴더의 new_*.ipynb 8개씩 순서대로 실행합니다.
   - 노트북당 최대 20분 타임아웃. SVM·LightGBM은 시간이 걸릴 수 있습니다.

(2) 각 폴더에서 수동 실행
   - 해당 폴더(예: 11. 연속형결측_평균대체)를 작업 디렉터리로 한 뒤
   - jupyter nbconvert --to notebook --execute --inplace new_로지스틱.ipynb
   - 같은 방식으로 new_경사하강법, new_KNN, new_SVM, new_랜덤포레스트, new_XGBoost, new_LightGBM, new_CatBoost 실행.

실행 후 각 폴더의 results/ 에 new_모델명.pkl 이 생성되면 종합_1-10번_전체모델비교.ipynb 에서 11~14번 결과까지 불러올 수 있습니다.
