"""
YOLO Model Training Script
"""

import os
import yaml
import argparse
import shutil
import glob
from pathlib import Path
from ultralytics import YOLO


def create_data_yaml(dataset_path, classes_file="classes.txt", output_file="data.yaml"):
    """
    Create YOLO data.yaml configuration file

    Args:
        dataset_path (str): Path to dataset directory
        classes_file (str): Name of classes file (default: classes.txt)
        output_file (str): Output YAML file name

    Returns:
        str: Path to created YAML file
    """
    dataset_path = Path(dataset_path)
    classes_txt_path = dataset_path / classes_file

    # Read classes.txt to get class names
    if not classes_txt_path.exists():
        print(f"Classes file not found at {classes_txt_path}")
        print("Please create a classes.txt file with one class name per line")
        return None

    with open(classes_txt_path, "r") as f:
        classes = []
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0:
                classes.append(line)

    if not classes:
        print("No classes found in classes.txt file!")
        return None

    number_of_classes = len(classes)
    print(f"Found {number_of_classes} classes: {classes}")

    # Create data dictionary with OS-independent paths
    data = {
        "path": str(dataset_path.absolute()),
        "train": str(Path("train") / "images"),
        "val": str(Path("validation") / "images"),
        "nc": number_of_classes,
        "names": classes,
    }

    # Write data to YAML file
    yaml_path = dataset_path / output_file
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"Created config file at {yaml_path}")

    # Display file contents
    print("\nYAML file contents:")
    with open(yaml_path, "r") as f:
        print(f.read())

    return str(yaml_path)


def train_yolo_model(
    data_yaml,
    model_name="yolo11s.pt",
    epochs=60,
    batch_size=16,
    img_size=640,
    device="auto",
):
    """
    Train YOLO model

    Args:
        data_yaml (str): Path to data.yaml config file
        model_name (str): YOLO model to use (default: yolo11s.pt)
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        img_size (int): Image size for training
        device (str): Device to use ('auto', 'cpu', 'cuda', etc.)

    Returns:
        str: Path to best model weights
    """
    print(f"\nStarting YOLO training...")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Device: {device}")

    # Initialize YOLO model
    model = YOLO(model_name)

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        save=True,
        verbose=True,
    )

    # Get path to best weights using OS-independent path
    best_weights = Path("runs") / "detect" / "train" / "weights" / "best.pt"

    if best_weights.exists():
        print(f"\nTraining completed! Best weights saved at: {best_weights}")
        return str(best_weights)
    else:
        print("Warning: Best weights file not found!")
        return None


def predict_and_save(model_weights, source_dir, output_dir="predictions"):
    """
    Run predictions on validation images

    Args:
        model_weights (str): Path to trained model weights
        source_dir (str): Directory containing images for prediction
        output_dir (str): Directory to save predictions
    """
    print(f"\nRunning predictions...")
    print(f"Model: {model_weights}")
    print(f"Source: {source_dir}")

    # Load trained model
    model = YOLO(model_weights)

    # Run predictions
    results = model.predict(
        source=source_dir, save=True, project=output_dir, name="predict"
    )

    # Show prediction results using OS-independent paths
    predict_dir = Path(output_dir) / "predict"
    if predict_dir.exists():
        image_files = list(predict_dir.glob("*.jpg"))[:5]  # Show first 5 images
        print(f"\nPrediction results saved to: {predict_dir}")
        print(f"Found {len(list(predict_dir.glob('*.jpg')))} prediction images")

        if image_files:
            print("Sample prediction files:")
            for img_path in image_files:
                print(f"  - {img_path.name}")


def save_model(weights_path, output_dir="models", model_name="trained_model"):
    """
    Save trained model and results

    Args:
        weights_path (str): Path to best weights
        output_dir (str): Output directory for saved model
        model_name (str): Name for saved model
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Copy best weights
    weights_path = Path(weights_path)
    if weights_path.exists():
        model_file = output_path / f"{model_name}.pt"
        shutil.copy2(weights_path, model_file)
        print(f"Model weights saved to: {model_file}")

    # Copy training results if available using OS-independent paths
    train_dir = Path("runs") / "detect" / "train"
    if train_dir.exists():
        results_dir = output_path / "training_results"
        if results_dir.exists():
            shutil.rmtree(results_dir)
        shutil.copytree(train_dir, results_dir)
        print(f"Training results saved to: {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for tile detection")
    parser.add_argument(
        "--dataset", default="dataset", help="Path to dataset directory"
    )
    parser.add_argument("--model", default="yolo11s.pt", help="YOLO model to use")
    parser.add_argument(
        "--epochs", type=int, default=60, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="Image size for training"
    )
    parser.add_argument(
        "--device", default="auto", help="Device to use (auto/cpu/cuda)"
    )
    parser.add_argument(
        "--output-name", default="tile_detection_model", help="Name for output model"
    )

    args = parser.parse_args()

    print("YOLO Training for Tile Detection")
    print("=" * 40)

    # Step 1: Create data.yaml config
    print("Step 1: Creating data configuration...")
    data_yaml = create_data_yaml(args.dataset)

    if not data_yaml:
        print("Failed to create data configuration. Exiting.")
        return

    # Step 2: Train model
    print("\nStep 2: Training YOLO model...")
    best_weights = train_yolo_model(
        data_yaml=data_yaml,
        model_name=args.model,
        epochs=args.epochs,
        img_size=args.img_size,
        device=args.device,
    )

    if not best_weights:
        print("Training failed. Exiting.")
        return

    # Step 3: Run predictions on validation set using OS-independent paths
    print("\nStep 3: Running predictions...")
    validation_images = Path(args.dataset) / "validation" / "images"
    if validation_images.exists():
        predict_and_save(str(best_weights), str(validation_images))
    else:
        print(f"Validation images not found at {validation_images}")

    # Step 4: Save final model
    print("\nStep 4: Saving model...")
    save_model(best_weights, output_dir="models", model_name=args.output_name)

    print("\nTraining pipeline completed!")
    print(f"Final model saved as: {Path('models') / f'{args.output_name}.pt'}")


if __name__ == "__main__":
    main()
