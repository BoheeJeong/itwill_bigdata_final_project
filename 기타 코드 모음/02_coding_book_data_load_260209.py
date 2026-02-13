import pandas as pd

from google.cloud import bigquery

file_path = r"C:\Users\bohee\Desktop\아이티윌\파이널\itwill_bigdata_final_project_code\raw\2022_light_version_data_mapping.csv"

df = pd.read_csv(file_path, encoding='utf-8')

key_path = "./itwill-final-gcp-key.json"
client = bigquery.Client.from_service_account_json(key_path)
print(f"연결된 프로젝트 ID: {client.project}")

project_id = f"{client.project}"
dataset_id = f"itwill_final_data"
table_id = "2022_codingbook_20260209_data"

dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")

dataset.location = "asia-northeast3"

try:
    dataset = client.create_dataset(dataset)
    print(f"데이터셋 {dataset.dataset_id}이(가) 생성되었습니다.")
except Exception as e:
    print(f"데이터셋 생성 중 오류 발생 (이미 존재할 수 있음): {e}")


# table = bigquery.Table(f"{project_id}.{dataset_id}.{table_id}")

job_config = bigquery.LoadJobConfig(
    # 테이블이 없으면 생성, 있으면 데이터 추가
    write_disposition="WRITE_APPEND", 
    # 데이터프레임의 헤더를 보고 스키마 자동 생성
    autodetect=True, 
)
job = client.load_table_from_dataframe(df, f"{project_id}.{dataset_id}.{table_id}", job_config=job_config)
job.result()  # 전송 완료까지 대기

query = f"""
    SELECT * 
    FROM `{project_id}.{dataset_id}.{table_id}`
  """
location = "asia-northeast3"


df_result = client.query(query).to_dataframe()
df_result
