import torch
import argparse


def load_checkpoint(checkpoint_path):
    """
    Load a checkpoint from the given path.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        dict: The loaded checkpoint containing model state, optimizer state, and other metadata.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        return checkpoint
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def print_checkpoint_keys(checkpoint):
    """
    Print all keys in the checkpoint and display performance values if available.

    Args:
        checkpoint (dict): The loaded checkpoint.
    """
    if checkpoint is None:
        print("No checkpoint to display.")
        return

    print("Checkpoint contains the following keys:")
    for key in checkpoint.keys():
        print(f"- {key}")
        if isinstance(checkpoint[key], dict):
            print("  Nested keys:")
            for sub_key in checkpoint[key].keys():
                print(f"  - {sub_key}")

    # Look for common performance-related keys
    performance_keys = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy', 'metrics']
    print("\nPerformance values in the checkpoint:")
    for key in performance_keys:
        if key in checkpoint:
            print(f"{key}: {checkpoint[key]}")


def print_checkpoint_details(checkpoint, indent=0):
    """
    Print detailed information about each key in the checkpoint.

    Args:
        checkpoint (dict): The loaded checkpoint.
        indent (int): The indentation level for nested keys.
    """
    if checkpoint is None:
        print("No checkpoint to display.")
        return

    indent_str = ' ' * indent
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            print(f"{indent_str}- {key} (dict):")
            print_checkpoint_details(value, indent=indent + 2)
        elif isinstance(value, (list, tuple)):
            print(f"{indent_str}- {key} ({type(value).__name__}, length {len(value)}):")
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    print(f"{indent_str}  - Index {i} (dict):")
                    print_checkpoint_details(item, indent=indent + 4)
                else:
                    print(f"{indent_str}  - Index {i}: {type(item).__name__}")
        else:
            print(f"{indent_str}- {key}: {value} ({type(value).__name__})")


def main(args):
    checkpoint = load_checkpoint(args.checkpoint_path)
    # print_checkpoint_keys(checkpoint)
    print_checkpoint_details(checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a checkpoint and print all keys.")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint file.")
    args = parser.parse_args()
    
    main(args)