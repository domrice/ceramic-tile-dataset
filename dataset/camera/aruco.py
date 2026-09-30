"""
ArUco marker detection and pose estimation.
"""

import numpy
import cv2
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple


class ArucoDetector:
    def __init__(self, marker_size_mm: float = 50.0):
        self.marker_size_mm = marker_size_mm

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

    @staticmethod
    def compute_scaling_factor(
        marker_poses: List[Dict], marker_size_mm: float = 50.0
    ) -> Tuple[Optional[float], Optional[Dict]]:
        """
        Compute mm/pixel scaling factor using ArUco marker edge lengths.

        Parameters
        ----------
        marker_poses : List[Dict]
            Detected markers with corner coordinates
        marker_size_mm : float
            Physical size of ArUco marker edges in millimeters (default: 50mm)

        Returns
        -------
        mm_per_pixel : float or None
            Median scaling factor (mm/pixel)
        quality_metrics : dict or None
            Quality metrics for the scaling computation
        """
        if not marker_poses:
            print("No markers detected!")
            return None, None

        print(f"\nComputing scaling factor from {len(marker_poses)} markers")

        edge_lengths_px = []

        # Compute all edge lengths in pixels
        for marker in marker_poses:
            pixel_corners = numpy.array(marker["corner_coordinates"])

            # Compute all 4 edges
            for i in range(4):
                j = (i + 1) % 4
                edge_length = numpy.linalg.norm(pixel_corners[i] - pixel_corners[j])

                if edge_length > 1.0:  # Valid measurement
                    edge_lengths_px.append(edge_length)

        if len(edge_lengths_px) < 4:
            print("Not enough valid edge measurements!")
            return None, None

        edge_lengths_px = numpy.array(edge_lengths_px)

        # Compute mm/pixel ratios for each edge
        ratios = marker_size_mm / edge_lengths_px

        # Use median for robustness against outliers
        mm_per_pixel = numpy.median(ratios)
        std_dev = numpy.std(ratios)

        quality_metrics = {
            "num_measurements": len(ratios),
            "num_markers": len(marker_poses),
            "marker_size_mm": marker_size_mm,
            "mm_per_pixel_median": float(mm_per_pixel),
            "mm_per_pixel_std": float(std_dev),
            "coefficient_of_variation": float(std_dev / mm_per_pixel),
            "mean_edge_length_px": float(numpy.mean(edge_lengths_px)),
            "std_edge_length_px": float(numpy.std(edge_lengths_px)),
        }

        return mm_per_pixel, quality_metrics

    def detect_markers(self, image: numpy.ndarray) -> Optional[List[Dict]]:
        """
        Detect ArUco markers in image and return their corner coordinates.

        Parameters
        ----------
        image : numpy.ndarray
            Input image (grayscale or BGR)

        Returns
        -------
        marker_data : List[Dict] or None
            List of detected markers with id and corner coordinates
        """
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return None

        marker_data = []

        for i, marker_id in enumerate(ids.flatten()):
            image_points = corners[i][0].astype(numpy.float32)

            marker_data.append(
                {
                    "id": int(marker_id),
                    "corner_coordinates": image_points.tolist(),
                }
            )

        return marker_data

    def save_markers(
        self,
        num_markers: int,
        output_dir: Path,
        marker_size_mm: int,
        dpi: int,
        border_bits: int,
    ) -> List[Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        print(f"Generating {num_markers} ArUco markers.")

        for marker_id in range(num_markers):
            marker_image = self._create_aruco_marker(
                marker_id, marker_size_mm, dpi, border_bits
            )

            filename = output_dir / f"aruco_{marker_id:03d}_{marker_size_mm}mm.png"
            cv2.imwrite(str(filename), marker_image)
            saved_files.append(filename)

        return saved_files

    def _create_aruco_marker(
        self, marker_id: int, marker_size_mm: int, dpi: int, border_bits: int
    ) -> None:
        marker_size_pixels = int((marker_size_mm / 25.4) * dpi)  # 1 inch = 25.4 mm

        print(f"Creating ArUco marker {marker_id}:")
        print(f"    Physical size: {marker_size_mm} mm")
        print(f"    Image size: {marker_size_pixels} px at {dpi} DPI")
        print(f"    Dictionary: DICT_7X7_250\n")

        marker_image = cv2.aruco.generateImageMarker(
            self.aruco_dict, marker_id, marker_size_pixels, border_bits
        )

        return marker_image
