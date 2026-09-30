"""
Main entry point for dataset collection pipeline.

Usage:
    python -m dataset [--interval 30] [--output data/captures]
"""

import numpy
import argparse
import time

from pathlib import Path

from .utils.config import Config
from .camera import CameraCapture
from .camera import ArucoDetector
from .segmentation import Segmentation
from .utils.tiledatabase import TileDatabase


class Colors:
    """
    ANSI color codes for terminal output
    """

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_status(message, status_type="info"):
    """Print colored status messages"""
    if status_type == "success":
        print(f"{Colors.GREEN}✓ {message}{Colors.END}")
    elif status_type == "warning":
        print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")
    elif status_type == "error":
        print(f"{Colors.RED}✗ {message}{Colors.END}")
    elif status_type == "info":
        print(f"{Colors.BLUE}ℹ {message}{Colors.END}")
    else:
        print(message)


def print_header(message):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE} {message}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def main():
    parser = argparse.ArgumentParser(description="Dataset collection pipeline")
    parser.add_argument(
        "--interval", type=int, default=30, help="Capture interval (seconds)"
    )
    parser.add_argument(
        "--output", type=Path, default=Config.CAPTURES_DIR, help="Output directory"
    )
    parser.add_argument(
        "--test",
        type=bool,
        default=False,
        help="True / False to take a test picture and detect ArUcos",
    )

    args = parser.parse_args()

    Config.ensure_dirs()
    # Initialize
    C = CameraCapture(ip=Config.IPHONE_IP, port=Config.IPHONE_PORT)
    D = ArucoDetector(marker_size_mm=50.0)
    S = Segmentation(
        yolo_model_path=Config.YOLO_MODEL,
        sam2_model_path=Config.SAM2_MODEL,
        yolo_conf_threshold=0.7,
    )

    # Health check
    if not C.health_check():
        print_status(f"Cannot reach camera!", "error")
        return

    if args.test:
        print_header("Starting Capturing Process Test")
        test_image = C.capture_test_image(timeout=True)

        img, K, dist_coeffs = C.capture_image()
        poses = D.detect_pose(image=img, K=K, dist_coeffs=dist_coeffs)

        if not poses:
            print_status("  No markers detected", "error")
        else:
            print_status("  Markers detected", "success")

        print_header("Finished Capturing Process Test")

    else:
        test_image = C.capture_test_image(timeout=False)
        print_status(f"Saving to {args.output}", "info")
        print_status(f"Capturing every {args.interval}s (CTRL+C to stop)\n", "info")

        print_header("Starting Capturing Process")
        filename = Config.TILES_DIR / "tiles_db.h5"
        with TileDatabase(db_path=filename, mask_shape=test_image.shape[:2]) as TDB:

            capture_count = 0
            try:
                while True:
                    print_status(
                        f"\nCapture {capture_count + 1} at {C.timestamp}", "info"
                    )

                    # 1. Capture
                    img, K, dist_coeffs = C.capture_image()

                    if not numpy.any(img):
                        print_status("Capture failed", "error")
                        continue

                    # 3. Detect ArUco
                    poses = D.detect_markers(image=img)

                    if not poses:
                        print_status("No markers detected, skipping", "warning")
                        continue

                    # 4. Compute scale
                    mm_per_pixel, quality = D.compute_scaling_factor(
                        poses, D.marker_size_mm
                    )

                    if mm_per_pixel is None:
                        print_status("Scale calculation failed, skipping", "warning")
                        continue

                    if quality["coefficient_of_variation"] > 0.1:
                        print_status(
                            f"Scale factor = {mm_per_pixel}, with {quality['mm_per_pixel_std']}",
                            "warning",
                        )
                        print_status(
                            f"High varation in scale factor ({quality['mm_per_pixel_std']} / {quality['mm_per_pixel_median']} = {quality['mm_per_pixel_std'] / quality['mm_per_pixel_median'] * 100:.1f}%)",
                            "warning",
                        )
                        print_status(
                            "Probable calibration or measurement errors", "warning"
                        )
                    else:
                        print_status(
                            f"Scale: {mm_per_pixel:.4f} mm/px (CV: {quality['coefficient_of_variation']*100:.1f}%)",
                            "info",
                        )
                        print_status(
                            f"Scale factor = {mm_per_pixel}, with {quality['mm_per_pixel_std']}",
                            "info",
                        )
                        print_status(
                            f"Mean Edge Length PX {quality['mean_edge_length_px']}, STD Edge Length PX {quality['std_edge_length_px']}",
                            "info",
                        )

                    time_start = time.time()

                    # 5. Segment
                    S.set_image(image=img)
                    segm_tiles = S.segment_by_points()

                    # 6. Store Tiles
                    masks = []
                    colors = []
                    for segm_tile in segm_tiles:
                        masks.append(segm_tile.mask)
                        colors.append(segm_tile.color)

                    TDB.add_tiles_batch(
                        masks=masks,
                        colors=colors,
                        scaling_factor=mm_per_pixel,
                        timestamp=C.timestamp,
                    )

                    end_time = time.time() - time_start
                    capture_count += 1
                    tile_count = len(TDB)

                    print_status(
                        f"Segmentation + Storage = {end_time:.2f} seconds", "info"
                    )
                    print_status(
                        f"Tile Count ({tile_count}) / Capture Count ({capture_count}) = {tile_count / capture_count} Tiles per Capture",
                        "info",
                    )
                    print_status(f"Waiting {args.interval}s...", "warning")
                    time.sleep(args.interval)

            except KeyboardInterrupt:
                print_status(f"\nStopped. {capture_count} captures saved.", "success")

                if len(TDB) > 0:
                    stats = TDB.get_stats()
                    print_header(f"\nFinal Statistics:")
                    print_status(f"    Total tiles: {stats['num_tiles']}", "info")
                    print_status(
                        f"    Database size: {stats['file_size_mb']:.1f} MB", "info"
                    )
                    print_status(
                        f"    Mean color: RGB {stats['color_distribution']['mean_rgb']}",
                        "info",
                    )
                    print_status(
                        f"    First capture: {stats['timestamps']['first']}", "info"
                    )
                    print_status(
                        f"    Last capture: {stats['timestamps']['last']}", "info"
                    )


if __name__ == "__main__":
    main()
