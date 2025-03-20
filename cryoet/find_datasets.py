import cryoet_data_portal as portal

# Instantiate a client
client = portal.Client()

# Query datasets with annotations containing 'mitochondria'
datasets = portal.Dataset.find(
    client, [portal.Dataset.runs.annotations.object_name.ilike("%mitochondria%")]
)

# Iterate over the datasets and print their IDs and associated annotations
for dataset in datasets:
    print(f"Dataset ID: {dataset.id}")
    for run in dataset.runs:
        for annotation in run.annotations:
            if 'mitochondria' in annotation.object_name.lower():
                print(f"  Annotation ID: {annotation.id}, Object Name: {annotation.object_name}")
