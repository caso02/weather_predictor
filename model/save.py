import os
import argparse
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

# Parse the Azure Storage connection string from command-line arguments
parser = argparse.ArgumentParser(description='Upload Models to Azure Blob Storage')
parser.add_argument('-c', '--connection', required=True, help="Azure Storage connection string")
args = parser.parse_args()

try:
    print("Azure Blob Storage Python quickstart sample")

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(args.connection)

    # Check existing containers and determine the next suffix
    exists = False
    containers = blob_service_client.list_containers(include_metadata=True)
    suffix = 0
    for container in containers:
        existing_container_name = container['name']
        print(existing_container_name, container['metadata'])
        if existing_container_name.startswith("weatherpredictor-model"):
            parts = existing_container_name.split("-")
            if len(parts) == 3:
                new_suffix = int(parts[-1])
                if new_suffix > suffix:
                    suffix = new_suffix

    suffix += 1
    container_name = f"weatherpredictor-model-{suffix}"
    print("New container name:")
    print(container_name)

    for container in containers:
        print("\t" + container['name'])
        if container_name in container['name']:
            print("Container already exists!")
            exists = True

    if not exists:
        # Create the container
        container_client = blob_service_client.create_container(container_name)

    # List of model files to upload
    model_files = [
        "temp_model.pkl",
        "weather_model.pkl",
        "weather_label_encoder.pkl"
    ]

    # Upload each model file to the container
    for local_file_name in model_files:
        upload_file_path = os.path.join("models", local_file_name)
        if not os.path.exists(upload_file_path):
            print(f"File {upload_file_path} does not exist. Skipping.")
            continue

        # Create a blob client using the local file name as the name for the blob
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)
        print(f"\nUploading to Azure Storage as blob:\n\t{local_file_name}")

        # Upload the file
        with open(file=upload_file_path, mode="rb") as data:
            blob_client.upload_blob(data)

except Exception as ex:
    print('Exception:')
    print(ex)
    exit(1)