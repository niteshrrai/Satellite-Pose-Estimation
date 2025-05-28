import os
import json
import random
import shutil
import argparse
import sys
from pathlib import Path


def organize_synthetic_images(base_path="synthetic"):
    """Splits image folder into train and val using their respective json files."""
    os.makedirs(os.path.join(base_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "val"), exist_ok=True)
    
    for split in ["train", "val"]:
        with open(os.path.join(base_path, f"{split}.json")) as f:
            files = {entry["filename"] for entry in json.load(f)}
        
        for file in files:
            shutil.move(os.path.join(base_path, "images", file), os.path.join(base_path, split, file))


def split_val_set(val_json, val_folder, split_ratio=0.5):
    """Splits the validation set into two halves: val and test."""
    random.seed(42)
    with open(val_json, 'r') as f:
        data = json.load(f)
    
    random.shuffle(data)
    split_point = int(len(data) * split_ratio)
    
    val_data = sorted(data[:split_point], key=lambda x: x['filename'])
    test_data = sorted(data[split_point:], key=lambda x: x['filename'])
    
    val_path = Path(val_folder)
    test_path = val_path.parent / "test"
    os.makedirs(test_path, exist_ok=True)
    
    for item in test_data:
        shutil.move(val_path / item['filename'], test_path / item['filename'])
    
    with open(val_json, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(test_path.parent / "test.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    return len(val_data), len(test_data)


def sample_subset(src_json, src_folder, dst_folder, total_count=None, train_ratio=0.8, val_ratio=0.1):
    """Creates a small subset of images with specified train/val/test split."""
    random.seed(42)
    with open(src_json, 'r') as f:
        data = json.load(f)
    
    random.shuffle(data)
    
    total_available = len(data)
    total_count = total_available if total_count is None else min(total_count, total_available)
    
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count
    
    train_data = sorted(data[:train_count], key=lambda x: x['filename'])
    val_data = sorted(data[train_count:train_count + val_count], key=lambda x: x['filename'])
    test_data = sorted(data[train_count + val_count:total_count], key=lambda x: x['filename'])
    
    subsets = {"train": train_data, "val": val_data, "test": test_data}
    
    for subset, subset_data in subsets.items():

        subset_dir = Path(dst_folder) / subset
        os.makedirs(subset_dir, exist_ok=True)
        
        for item in subset_data:
            shutil.copy2(Path(src_folder) / item['filename'], subset_dir / item['filename'])
        
        with open(Path(dst_folder) / f"{subset}.json", 'w') as f:
            json.dump(subset_data, f, indent=2)
    
    return train_count, val_count, test_count


def consolidate_images(base_path):
    images_path = Path(base_path) / "images"
    os.makedirs(images_path, exist_ok=True)
    
    for subset in ["train", "val", "test"]:
        src, dst = Path(base_path) / subset, images_path / subset
        if src.exists():
            os.makedirs(dst, exist_ok=True)
            for file in src.iterdir():
                if file.is_file():
                    shutil.move(str(file), str(dst / file.name))
            try:
                os.rmdir(src)
            except OSError:
                print(f"Skipped deleting {src}, as it is not empty.")


def main():
    parser = argparse.ArgumentParser(description='Sample data and split validation set')
    parser.add_argument('--src', type=str, required=True, help='Source directory')
    parser.add_argument('--dst', type=str, required='--subset' in sys.argv, help='Destination directory')
    parser.add_argument('--json', type=str, required=True, help='Source JSON file')
    parser.add_argument('--images', type=str, required=True, help='Source images directory')
    parser.add_argument('--count', type=int, help='Total number of images to sample for the subset')
    parser.add_argument('--subset', action='store_true', help='Create a subset or divide the data')
    parser.add_argument('--split-val', action='store_true', help='Split existing val set into val and test')
    
    args = parser.parse_args()
    
    if args.subset:
        train_count, val_count, test_count = sample_subset(
            src_json=Path(args.src) / args.json,
            src_folder=Path(args.src) / "images" / args.images,
            dst_folder=args.dst,
            total_count=args.count,
        )
        print(f"Subset created: {train_count} train, {val_count} val, {test_count} test images")
    
    if args.split_val:
        val_count, test_count = split_val_set(
            val_json=Path(args.src) / "val.json",
            val_folder=Path(args.src) / "val",
        )
        print(f"Validation set split: {val_count} val, {test_count} test images")


if __name__ == "__main__":
    main()

# Example Usage:
# python3 data_preprocessing.py --src synthetic --json val.json --images val --split-val (don't run it again)
# python3 data_preprocessing.py --src synthetic --dst cyclegantest --json train.json --images train --subset --count 1000 


