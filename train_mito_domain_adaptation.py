import argparse
import os
from glob import glob

from sklearn.model_selection import train_test_split
from synaptic_reconstruction.training.domain_adaptation import mean_teacher_adaptation

from config import SAVE_DIR

ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held"
NAMES = ["vesicle_pools", "tether", "rat"]


def _get_paths(root):
    paths = sorted(glob(os.path.join(root, "**", "*.h5")))
    return paths


def run_structure_domain_adaptation(args):
    paths = _get_paths(args.data_dir)
    train_paths, val_paths = train_test_split(paths, test_size=0.15, random_state=42)
    patch_shape = (64, 512, 512)
    mean_teacher_adaptation(
        name=args.experiment_name,
        unsupervised_train_paths=train_paths,
        unsupervised_val_paths=val_paths,
        patch_shape=patch_shape,
        save_root=SAVE_DIR,
        source_checkpoint=args.checkpoint_path_teacher,
    )


def main():
    parser = argparse.ArgumentParser(description="3D UNet domain adaptation mean teacher for mitochondrial segmentation")
    parser.add_argument("--experiment_name", type=str, default="mito-domain-adapt", help="Name that is used for the experiment and store the model's weights")
    parser.add_argument("--data_dir", type=str, default="/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held", help="Path to the data directory")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=(64, 512, 512), help="Patch shape for data loading (3D tuple)")
    parser.add_argument("--checkpoint_path", required=True, type=str, default="", help="Path to checkpoint used to load model's state_dict for teacher")
    args = parser.parse_args()
    
    run_structure_domain_adaptation(args)
    # for name in NAMES:
    #     run_structure_domain_adaptation(name)


if __name__ == "__main__":
    main()