import datetime
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

with DAG(
    dag_id="debug_kubectl_version_simple",
    start_date=datetime.datetime(2024, 1, 1),
    schedule=None,  # 設定為 None，手動觸發即可
    catchup=False,
    tags=["debug", "kubectl", "maintenance"],
    description="A simple DAG to check kubectl client version on the worker.",
) as dag:
  # 直接使用 BashOperator，完全不依賴專案內的 Utils
  check_version = BashOperator(
      task_id="check_kubectl_client_version",
      # --client: 只檢查客戶端版本 (不需要 Cluster 憑證)
      # --output=yaml: 輸出成易讀的格式
      bash_command="kubectl version",
  )

  check_version
