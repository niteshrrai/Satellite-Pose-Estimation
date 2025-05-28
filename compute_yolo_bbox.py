import json
import os
import argparse
from pathlib import Path
from PIL import Image


def convert_bbox_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
    """Converts bounding box from [x_min, y_min, x_max, y_max] to YOLO format [x_center, y_center, width, height]"""
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return x_center, y_center, width, height


def main():
    parser = argparse.ArgumentParser(description='Convert JSON bounding boxes to YOLO format')
    parser.add_argument('--src', type=str, required=True, help='Source directory')
    parser.add_argument('--json', type=str, required=True, help='JSON filename')
    parser.add_argument('--images', type=str, required=True, help='Images directory')
    parser.add_argument('--class_id', type=int, default=0, help='Class ID for the object (default: 0)')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True, help='Dataset split (train val or test)')
    args = parser.parse_args()
    
    json_path = Path(args.src) / args.json
    with open(json_path, 'r') as f:
        data = json.load(f)

    output_dir = Path(args.src) / "labels"
    label_dir = output_dir / args.split
    label_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {args.split} data. Labels will be saved to: {label_dir}")
    
    image_dir = Path(args.src) / "images" /args.images     
    
    for entry in data:
        try:
            filename = entry['filename']
            bbox = entry['bbox_gt']
            
            base_filename = os.path.splitext(filename)[0]

            image_path = image_dir / filename
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except FileNotFoundError:
                print(f"Warning: Image file {filename} not found, skipping...")
                continue

            x_min, y_min, x_max, y_max = bbox
            x_center, y_center, width, height = convert_bbox_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height)
            
            label_path = label_dir / f"{base_filename}.txt"

            with open(label_path, 'w') as f:
                f.write(f"{args.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
        except (KeyError, IndexError) as e:
            print(f"Error processing entry for {entry.get('filename', 'unknown')}: {e}")
    
    print(f"Conversion complete. YOLO format labels saved to {label_dir}")

if __name__ == "__main__":
    main()


# Example Usage:
# python3 compute_yolo_bbox.py --src synthetic --json train.json --images train  --class_id 0 --split train
# python3 compute_yolo_bbox.py --src synthetic --json val.json --images val  --class_id 0 --split val
# python3 compute_yolo_bbox.py --src synthetic --json test.json --images test  --class_id 0 --split test
