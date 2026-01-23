import datetime
import json
from airflow import DAG
from airflow.decorators import task
import google.auth
from googleapiclient import discovery
from airflow.models import Variable


PUBLIC_KEY_CONTENT = Variable.get("jax_shared_public_key_v2")
SA_EMAIL = "ml-auto-solutions@cloud-ml-auto-solutions.iam.gserviceaccount.com"


default_args = {
    'owner': 'airflow',
    'start_date': datetime.datetime(2025, 1, 1),
}

with DAG(
    'a_0_setup_oslogin_and_verify',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    @task
    def self_register_and_list_keys():
        credentials, project = google.auth.default()
        service = discovery.build('oslogin', 'v1', credentials=credentials)

        parent = f'users/{SA_EMAIL}'

        print(f"Registering persistent SSH key for: {SA_EMAIL}...")
        import_body = {'key': PUBLIC_KEY_CONTENT}
        import_response = service.users().importSshPublicKey(
            parent=parent,
            body=import_body
        ).execute()

        print("--- SSH Key Import Successful---")

        profile = service.users().getLoginProfile(name=parent).execute()

        keys = profile.get('sshPublicKeys', {})
        print(f"Found {len(keys)} key(s) in the profile:")

        for key_id, key_data in keys.items():
            # Check if this is the key we just added (by matching the comment/tag)
            # Make sure your PUBLIC_KEY_CONTENT ends with "airflow-shared-key"
            is_target = "airflow-shared-key" in key_data.get('key', '')
            status = "[PERSISTENT KEY]" if is_target else "[EPHEMERAL/OTHER KEY]"
            print(f"Key ID: {key_id}")
            print(f"Status: {status}")
            print(f"Fingerprint: {key_data.get('fingerprint')}")
            print("-" * 30)

        if any("airflow-shared-key" in k.get('key', '') for k in keys.values()):
            print("SUCCESS: Persistent SSH key is confirmed in the OS Login profile!")
        else:
            print("WARNING: Target key not found in profile. Please verify PUBLIC_KEY_CONTENT format.")

    self_register_and_list_keys()
