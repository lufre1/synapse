import napari

from cryoet_data_portal import Client, Tomogram, Annotation, Alignment, AnnotationFile, Dataset

client = Client()

tomograms = []
datasets = Dataset.find(client, [Dataset.runs.annotations.object_name.ilike("%mitochondrion%")])

# Iterate over all datasets
for dataset in datasets:
    print(f"Dataset: {dataset.title}")
    for run in dataset.runs:
        print(len(tomograms))
        if len(tomograms) > 10:
            break
        print(f"  - run: {run.name}")
        for tomo in run.tomograms:
            print(f"    - tomo: {tomo.name}")
            tomograms.append(tomo)




for tomo in tomograms:
    print(tomo.name)
    print(tomo.size_z)
    url = tomo.https_omezarr_dir
    viewer = napari.Viewer()
    viewer.open(url, plugin="napari-ome-zarr")
    break

napari.run()
