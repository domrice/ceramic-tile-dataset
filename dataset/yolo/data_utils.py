"""
Data preparation utilities for YOLO training
"""

import os
import shutil
import random
from pathlib import Path
import argparse


def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8, val_ratio=0.2):
    """
    Split dataset into training and validation sets

    Args:
        images_dir (str): Directory containing images
        labels_dir (str): Directory containing label files
        output_dir (str): Output directory for split dataset
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)

    # Get all image files
    image_files = (
        list(images_path.glob("*.jpg"))
        + list(images_path.glob("*.png"))
        + list(images_path.glob("*.jpeg"))
    )

    if not image_files:
        print(f"No image files found in {images_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Shuffle the files
    random.shuffle(image_files)

    # Calculate split indices
    total_files = len(image_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)

    # Adjust if ratios don't add up to 1.0
    if train_count + val_count > total_files:
        val_count = total_files - train_count

    print(f"Training files: {train_count}")
    print(f"Validation files: {val_count}")

    # Create output directories
    train_img_dir = output_path / "train" / "images"
    train_lbl_dir = output_path / "train" / "labels"
    val_img_dir = output_path / "validation" / "images"
    val_lbl_dir = output_path / "validation" / "labels"

    for dir_path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Split files
    for i, img_file in enumerate(image_files):
        # Determine which set this file goes to
        if i < train_count:
            img_dest_dir = train_img_dir
            lbl_dest_dir = train_lbl_dir
        elif i < train_count + val_count:
            img_dest_dir = val_img_dir
            lbl_dest_dir = val_lbl_dir
        else:
            continue  # Skip remaining files if any

        # Copy image file
        img_dest = img_dest_dir / img_file.name
        shutil.copy2(img_file, img_dest)

        # Copy corresponding label file
        label_file = labels_path / f"{img_file.stem}.txt"
        if label_file.exists():
            lbl_dest = lbl_dest_dir / label_file.name
            shutil.copy2(label_file, lbl_dest)
        else:
            print(f"Warning: No label file found for {img_file.name}")

    print(f"Dataset split completed! Output saved to: {output_dir}")


def create_classes_file(class_names, output_file="classes.txt"):
    """
    Create classes.txt file with class names

    Args:
        class_names (list): List of class names
        output_file (str): Output file path
    """
    with open(output_file, "w") as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

    print(f"Created classes file: {output_file}")
    print(f"Classes: {class_names}")


def validate_dataset(dataset_dir):
    """
    Validate dataset structure and check for missing files

    Args:
        dataset_dir (str): Path to dataset directory
    """
    dataset_path = Path(dataset_dir)

    print(f"Validating dataset: {dataset_dir}")
    print("-" * 40)

    # Check directory structure
    required_dirs = [
        "train/images",
        "train/labels",
        "validation/images",
        "validation/labels",
    ]

    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.glob("*")))
            print(f"✓ {dir_name}: {file_count} files")
        else:
            print(f"✗ {dir_name}: Missing!")

    # Check for classes.txt
    classes_file = dataset_path / "classes.txt"
    if classes_file.exists():
        with open(classes_file, "r") as f:
            classes = [line.strip() for line in f if line.strip()]
        print(f"✓ classes.txt: {len(classes)} classes - {classes}")
    else:
        print("✗ classes.txt: Missing!")

    # Check for matching image and label files
    print("\nChecking for missing labels...")

    for split in ["train", "validation"]:
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"

        if img_dir.exists() and lbl_dir.exists():
            img_files = (
                list(img_dir.glob("*.jpg"))
                + list(img_dir.glob("*.png"))
                + list(img_dir.glob("*.jpeg"))
            )
            missing_labels = []

            for img_file in img_files:
                label_file = lbl_dir / f"{img_file.stem}.txt"
                if not label_file.exists():
                    missing_labels.append(img_file.name)

            if missing_labels:
                print(f"✗ {split}: {len(missing_labels)} images missing labels")
                if len(missing_labels) <= 5:
                    for missing in missing_labels:
                        print(f"    - {missing}")
                else:
                    print(
                        f"    - {missing_labels[0]} ... and {len(missing_labels)-1} more"
                    )
            else:
                print(f"✓ {split}: All images have corresponding labels")


def main():
    parser = argparse.ArgumentParser(description="YOLO dataset utilities")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Split dataset command
    split_parser = subparsers.add_parser(
        "split", help="Split dataset into train/validation"
    )
    split_parser.add_argument(
        "--images", required=True, help="Directory containing images"
    )
    split_parser.add_argument(
        "--labels", required=True, help="Directory containing labels"
    )
    split_parser.add_argument("--output", required=True, help="Output directory")
    split_parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Training ratio"
    )
    split_parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Validation ratio"
    )

    # Create classes command
    classes_parser = subparsers.add_parser("classes", help="Create classes.txt file")
    classes_parser.add_argument("--names", nargs="+", required=True, help="Class names")
    classes_parser.add_argument("--output", default="classes.txt", help="Output file")

    # Validate dataset command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate dataset structure"
    )
    validate_parser.add_argument("--dataset", required=True, help="Dataset directory")

    args = parser.parse_args()

    if args.command == "split":
        split_dataset(
            images_dir=args.images,
            labels_dir=args.labels,
            output_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

    elif args.command == "classes":
        create_classes_file(args.names, args.output)

    elif args.command == "validate":
        validate_dataset(args.dataset)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
