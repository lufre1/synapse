import argparse
import os
import hashlib
from collections import defaultdict

from tqdm import tqdm


def get_file_hash(file_path, chunk_size=8192):
    """Compute the SHA256 hash of a file in chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_valid_duplicates(base_path):
    """
    Find files with the same name in different directories and check if they are true duplicates.

    Parameters:
        base_path (str): Root directory to search.

    Returns:
        duplicates (dict): A dictionary mapping filenames to lists of duplicate file paths.
    """
    file_map = defaultdict(list)  # Maps filename to list of paths
    duplicates = defaultdict(list)  # Maps filename to list of true duplicate paths

    # Step 1: Collect all files and group by name
    for root, _, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_map[file].append(file_path)

    # Step 2: Check for valid duplicates by comparing file hashes
    for filename, paths in tqdm(file_map.items()):
        if len(paths) > 1:  # Only process files with duplicate names
            hash_map = {}  # Maps file hash to file path

            for path in paths:
                file_hash = get_file_hash(path)

                if file_hash in hash_map:
                    # Found a valid duplicate
                    if filename not in duplicates:
                        duplicates[filename].append(hash_map[file_hash])  # Add original file
                    duplicates[filename].append(path)
                else:
                    hash_map[file_hash] = path

    return duplicates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/cooper/", help="Path to the root data directory")
    # parser.add_argument("--export_path", "-e", type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/cooper/s2/", help="Path to the export directory")
    # parser.add_argument("--scale_factor", "-s", type=int, default=2, help="Scale factor for the image")
    args = parser.parse_args()
    
    # Example usage:
    base_path = args.base_path
    duplicates = find_valid_duplicates(base_path)

    # Print results
    for filename, files in duplicates.items():
        print(f"\nValid duplicates for '{filename}':")
        for f in files:
            print(f" - {f}")


if __name__ == "__main__":
    main()