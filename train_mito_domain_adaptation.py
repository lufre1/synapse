import argparse
import os
from glob import glob
import random
import torch

from sklearn.model_selection import train_test_split
from synaptic_reconstruction.training.domain_adaptation import mean_teacher_adaptation

from config import SAVE_DIR


ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held"
NAMES = ["vesicle_pools", "tether", "rat"]


def _get_paths(root):
    paths = sorted(glob(os.path.join(root, "**", "*.h5")))
    return paths


def sampler_func(pseudo_labels, label_filter, threshold=0.75, min_fraction=0.1, p=0.95):
    sampled_labels = pseudo_labels[label_filter.to(torch.bool)]
    foreground_prediction = (sampled_labels > threshold).sum()
    foreground_fraction = foreground_prediction / label_filter.sum() 
    if foreground_fraction > min_fraction:
        return True
    else:
        return random.random() > p
        

def run_structure_domain_adaptation(args):
    paths = _get_paths(args.data_dir)
    train_paths, val_paths = train_test_split(paths, test_size=0.15, random_state=42)
    train_paths, test_paths = train_test_split(train_paths, test_size=0.10, random_state=42)
    print(f"train_paths {len(train_paths)} val_paths {len(val_paths)} test_paths {len(test_paths)}")
    print(f"all test_paths {test_paths}")
    sampler = sampler_func
    print(f"\n {args.experiment_name} \n")
    mean_teacher_adaptation(
        name=args.experiment_name,
        unsupervised_train_paths=train_paths,
        unsupervised_val_paths=val_paths,
        patch_shape=args.patch_shape,
        batch_size=args.batch_size,
        n_iterations=args.n_iterations,
        lr=args.learning_rate,
        save_root=SAVE_DIR,
        source_checkpoint=args.checkpoint_path,
        sampler=sampler,
        confidence_threshold=args.confidence_threshold,
        #early_stopping=3,
    )


def main():
    parser = argparse.ArgumentParser(description="3D UNet domain adaptation mean teacher for mitochondrial segmentation")
    parser.add_argument("--experiment_name", type=str, default="mito-domain-adapt", help="Name that is used for the experiment and store the model's weights")
    parser.add_argument("--data_dir", type=str, default="/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held_s2", help="Path to the data directory")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=(32, 256, 256), help="Patch shape for data loading (3D tuple)")
    parser.add_argument("--checkpoint_path", required=True, type=str, default="", help="Path to checkpoint used to load model's state_dict for teacher")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to be used")
    parser.add_argument("--n_iterations", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--confidence_threshold", type=float, default=0.75, help="Confidence threshold for pseudo-labeling")
    args = parser.parse_args()
    
    run_structure_domain_adaptation(args)
    # for name in NAMES:
    #     run_structure_domain_adaptation(name)


if __name__ == "__main__":
    main()
