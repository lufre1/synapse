import cryoet_data_portal as cdp
import s3fs
import os
from tqdm import tqdm

# S3 filesystem instance
fs = s3fs.S3FileSystem(anon=True)

# Client instance
client = cdp.Client()

# Run IDs (integers)
runs = [16498, 16514, 16581, 16641]

root = "/home/freckmann15/data/cryo-et/from_portal"

# Loop over run IDs
for run_id in tqdm(runs):
    # Query denoised tomograms
    tomograms = cdp.Tomogram.find(
        client,
        [
            cdp.Tomogram.run_id == run_id,
            cdp.Tomogram.processing == "denoised",
        ]
    )

    # Select the first tomogram (there should only be one in this case)
    tomo = tomograms[0]

    # Download the denoised tomogram
    output_folder = os.path.join(root, str(run_id))
    os.makedirs(output_folder, exist_ok=True)
    fname = f"{tomo.id}_{tomo.processing}.mrc"
    output_path = os.path.join(output_folder, fname)
    if os.path.exists(output_path):
        continue
    fs.get(tomo.s3_mrc_file, output_path)
