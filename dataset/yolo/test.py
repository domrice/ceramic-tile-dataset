"""
YOLO Model Testing and Inference Script
For tile detection
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def test_model(
    model_path,
    test_images,
    confidence=0.5,
    save_results=True,
    output_dir="test_results",
):
    """
    Test trained YOLO model on images

    Args:
        model_path (str): Path to trained model weights
        test_images (str): Path to test images directory or single image
        confidence (float): Confidence threshold for detections
        save_results (bool): Whether to save annotated images
        output_dir (str): Directory to save results

    Returns:
        list: Detection results
    """
    print(f"Loading model from: {model_path}")

    # Load trained model
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return None

    model = YOLO(str(model_path))

    # Create output directory
    if save_results:
        Path(output_dir).mkdir(exist_ok=True)

    # Get test images
    test_path = Path(test_images)
    if test_path.is_file():
        image_files = [test_path]
    elif test_path.is_dir():
        image_files = (
            list(test_path.glob("*.jpg"))
            + list(test_path.glob("*.png"))
            + list(test_path.glob("*.jpeg"))
        )
    else:
        print(f"Test path not found: {test_images}")
        return None

    if not image_files:
        print("No image files found!")
        return None

    print(f"Found {len(image_files)} test images")

    all_results = []

    for i, img_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {img_path.name}")

        # Run inference
        results = model(str(img_path), conf=confidence)

        # Process results
        for result in results:
            # Get detection info
            boxes = result.boxes
            if boxes is not None:
                detections = []
                for box in boxes:
                    # Get box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = model.names[cls]

                    detections.append(
                        {
                            "image": img_path.name,
                            "class": class_name,
                            "confidence": float(conf),
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        }
                    )

                all_results.extend(detections)

                # Print detections
                if detections:
                    print(f"  Found {len(detections)} objects:")
                    for det in detections:
                        print(f"    - {det['class']}: {det['confidence']:.2f}")
                else:
                    print("  No objects detected")

            # Save annotated image
            if save_results:
                annotated_img = result.plot()
                output_path = Path(output_dir) / f"annotated_{img_path.name}"
                cv2.imwrite(str(output_path), annotated_img)
                print(f"  Saved annotated image: {output_path}")

    # Print summary
    print(f"\nTesting completed!")
    print(f"Total detections: {len(all_results)}")
    if save_results:
        print(f"Annotated images saved to: {output_dir}")

    return all_results


def evaluate_model(model_path, dataset_yaml):
    """
    Evaluate model performance on validation dataset

    Args:
        model_path (str): Path to trained model weights
        dataset_yaml (str): Path to dataset YAML configuration
    """
    print(f"Evaluating model: {model_path}")
    print(f"Dataset config: {dataset_yaml}")

    model_path = Path(model_path)
    dataset_yaml = Path(dataset_yaml)

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    if not dataset_yaml.exists():
        print(f"Dataset config not found: {dataset_yaml}")
        return

    # Load model
    model = YOLO(str(model_path))

    # Run evaluation
    results = model.val(data=str(dataset_yaml))

    # Print metrics
    print("\nEvaluation Results:")
    print(f"mAP50: {results.box.map50:.3f}")
    print(f"mAP50-95: {results.box.map:.3f}")

    return results


def real_time_detection(model_path, source=0, confidence=0.5):
    """
    Real-time detection using webcam or video stream

    Args:
        model_path (str): Path to trained model weights
        source: Video source (0 for webcam, path to video file)
        confidence (float): Confidence threshold
    """
    print(f"Starting real-time detection...")
    print(f"Model: {model_path}")
    print(f"Source: {source}")
    print("Press 'q' to quit")

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    # Load model
    model = YOLO(str(model_path))

    # Start video capture
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame, conf=confidence)

        # Draw results
        annotated_frame = results[0].plot()

        # Display frame
        cv2.imshow("YOLO Detection", annotated_frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Real-time detection stopped")


def main():
    parser = argparse.ArgumentParser(description="Test YOLO model for tile detection")
    parser.add_argument("--model", required=True, help="Path to trained model weights")
    parser.add_argument(
        "--images", help="Path to test images directory or single image"
    )
    parser.add_argument("--dataset", help="Path to dataset YAML for evaluation")
    parser.add_argument(
        "--realtime", action="store_true", help="Run real-time detection"
    )
    parser.add_argument(
        "--source", default=0, help="Video source for real-time detection"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Confidence threshold"
    )
    parser.add_argument(
        "--output", default="test_results", help="Output directory for results"
    )

    args = parser.parse_args()

    print("YOLO Model Testing")
    print("=" * 30)

    if args.realtime:
        # Real-time detection
        real_time_detection(args.model, args.source, args.confidence)

    elif args.images:
        # Test on images
        results = test_model(
            model_path=args.model,
            test_images=args.images,
            confidence=args.confidence,
            output_dir=args.output,
        )

        if results:
            print(f"\nDetection results saved to: {args.output}")

    elif args.dataset:
        # Evaluate on dataset
        evaluate_model(args.model, args.dataset)

    else:
        print("Please specify --images, --dataset, or --realtime")
        print("Use --help for more information")


if __name__ == "__main__":
    main()
