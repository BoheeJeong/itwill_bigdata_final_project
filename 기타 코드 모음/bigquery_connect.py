from google.cloud import bigquery
import pandas as pd

key_path = "./bigqurey_google_account.json"
client = bigquery.Client.from_service_account_json(key_path)

print(f"연결된 프로젝트 ID: {client.project}")

project_id = "gyehong-test"
dataset_id = f"{project_id}.api_connect_dataset"


dataset = bigquery.Dataset(dataset_id)

# 데이터셋 위치 설정 (서울은 'asia-northeast3', 기본은 'US')
dataset.location = "asia-northeast3"

# 2. 데이터셋 생성 요청
try:
    dataset = client.create_dataset(dataset)
    print(f"데이터셋 {dataset.dataset_id}이(가) 생성되었습니다.")
except Exception as e:
    print(f"데이터셋 생성 중 오류 발생 (이미 존재할 수 있음): {e}")


# 1. 테이블 ID 설정 (프로젝트.데이터셋.테이블)
table_id = f"{dataset_id}.bh_table"

# 2. 스키마(컬럼 구성) 정의
schema = [
    bigquery.SchemaField("user_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("user_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("signup_date", "DATE", mode="NULLABLE"),
    bigquery.SchemaField("is_active", "BOOLEAN", mode="NULLABLE"),
]

# 3. 테이블 객체 생성 및 API 요청
table = bigquery.Table(table_id, schema=schema)

# --- 파티셔닝 설정 추가 ---
# table.time_partitioning = bigquery.TimePartitioning(
#     type_=bigquery.TimePartitioningType.DAY, # 일 단위로 파티션
#     field="signup_date"                      # 파티션 기준이 될 컬럼명
# )

# 실수로 전체 테이블을 스캔하는 것을 방지하고 싶다면 (권장)
# table.require_partition_filter = True


try:
    table = client.create_table(table)
    print(f"테이블 {table.table_id}이(가) 생성되었습니다.")
except Exception as e:
    print(f"테이블 생성 중 오류 발생: {e}")


data = {
    "user_id": [3, 4, 5],
    "user_name": ["Charlie", "David", "Eve"],
    "signup_date": ["2026-02-01", "2026-02-05", "2026-02-07"],
    "is_active": [True, True, False]
}
df = pd.DataFrame(data)

# 데이터타입 보정 (날짜 형식을 빅쿼리 DATE에 맞게 변환)
df['signup_date'] = pd.to_datetime(df['signup_date']).dt.date


# 4. 로드 설정 (데이터를 덮어쓸지, 추가할지 결정)
job_config = bigquery.LoadJobConfig(
    # WRITE_APPEND: 기존 데이터 뒤에 추가 (기본값)
    # WRITE_TRUNCATE: 기존 데이터 삭제 후 새로 쓰기
    write_disposition="WRITE_APPEND", 
)

# 5. 데이터프레임 전송
job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
job.result()  # 전송 완료까지 대기

print(f"성공적으로 {len(df)}건의 데이터가 {table_id}에 삽입되었습니다.")

# 가장 간단하게 테이블 전체를 DataFrame으로 가져오기
query = f"SELECT * FROM `{table_id}` ORDER BY user_id"




df_result = client.query(query).to_dataframe()

print("\n--- 현재 테이블 전체 데이터 ---")
print(df_result)


create_table("gyehong-test", "test_dataset", "bohee1")


def create_table(project_id, dataset_id, table_id, schema=None):
    # 1. 테이블 ID 설정 (프로젝트.데이터셋.테이블)
    table_id = f"{project_id}.{dataset_id}.{table_id}"

    # 2. 스키마(컬럼 구성) 정의
    schema = [
        bigquery.SchemaField("user_id", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("user_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("signup_date", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("is_active", "BOOLEAN", mode="NULLABLE"),
    ]

    # 3. 테이블 객체 생성 및 API 요청
    table = bigquery.Table(table_id, schema=schema)

    # --- 파티셔닝 설정 추가 ---
    # table.time_partitioning = bigquery.TimePartitioning(
    #     type_=bigquery.TimePartitioningType.DAY, # 일 단위로 파티션
    #     field="signup_date"                      # 파티션 기준이 될 컬럼명
    # )

    # 실수로 전체 테이블을 스캔하는 것을 방지하고 싶다면 (권장)
    # table.require_partition_filter = True


    try:
        table = client.create_table(table)
        print(f"테이블 {table.table_id}이(가) 생성되었습니다.")
    except Exception as e:
        print(f"테이블 생성 중 오류 발생: {e}")


project_id = "gyehong-test"
dataset_id = "api_connect_dataset"

create_table(project_id, dataset_id, "bh5")

def insert_table(prject_id, dataset_id, table_id, df, job_config):
   
    full_path = f"{project_id}.{dataset_id}.{table_id}"
    df = df

    # 4. 로드 설정 (데이터를 덮어쓸지, 추가할지 결정)
    job_config = job_config

    # 5. 데이터프레임 전송
    job = client.load_table_from_dataframe(df, full_path, job_config=job_config)
    job.result()  # 전송 완료까지 대기

    print(f"성공적으로 {len(df)}건의 데이터가 {table_id}에 삽입되었습니다.")



data = {
        "user_id": [3, 4, 5],
        "user_name": ["Charlie", "David", "Eve"],
        "signup_date": ["2026-02-01", "2026-02-05", "2026-02-07"],
        "is_active": [True, True, False]
    }
df = pd.DataFrame(data)
# 데이터타입 보정 (날짜 형식을 빅쿼리 DATE에 맞게 변환)
df['signup_date'] = pd.to_datetime(df['signup_date']).dt.date
job_config = bigquery.LoadJobConfig(
        # WRITE_APPEND: 기존 데이터 뒤에 추가 (기본값)
        # WRITE_TRUNCATE: 기존 데이터 삭제 후 새로 쓰기
        write_disposition="WRITE_APPEND", 
    )

insert_table(project_id, dataset_id, "bh5", df, job_config)