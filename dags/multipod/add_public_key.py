import datetime
import json
from airflow import DAG
from airflow.decorators import task
import google.auth
from googleapiclient import discovery


PUBLIC_KEY_CONTENT = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCs8oH4ZnW9OP//3qz4aLjON86HO6drJz7xj09ofR419TH5AzsAnFql8vMbxwT9sVC+usqgJdk+NSIBIlToB0wXJeWI5OnFIj/aXVYDEmWm22M5djzY5EthN7gXXuDPpKt1bT3R9ol8Aa8Epbkod18yPKrY1TaiApqD6mH0w37B8eZmgb3DQSo97wFbYhYzYYz/1cS2BFVS8HkDSPHFtESF6OlCGLF7pJPcdsah/sWzeRUF3Ly6dJfUVm01+YXyzRqWo99f0958nDg46mFgqtzkK+1bnZZVE3pYfOZhKuG1eShWWire1rPi8667bhTCKJ0+Z5WLpJnCMTFW0NsCfrpuJZWEzuYQy9K7Swd+ZJPh0UKIPaQSnMKeqCmK2pE2nch7ijNBPCF88GSOleEL0KJCGgQKHNoujRWvj21QsFY229Bs4zxv3+3JBXst+2wtOprdc2gN0Fi7ncEdT3rNK6MPi4m9tYbUui6af7VzUpBnBCckTpFC83rAFBiaCuUvq2dlqzIPqZopseZ8sXigGLu4D9Jg45W70CPUuB/RVoEbiiNtyLWqYHBIB3c1FDXWiOOVQuaWCP4kjpQSO83dz7kcQWy/5lTRmd2t4zDS1XpRxpfVmbGaW7j3YvdnfYey/AbO87r5EjPEqoVyOid6qcvk8tCOKi4BnPWMR9EF/R3JVw== airflow-shared-key"
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
