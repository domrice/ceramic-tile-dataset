"""
Configuration management for the pipeline.
"""

from cv2 import resize as cv2_resize
from cv2 import INTER_AREA
from cv2 import INTER_LANCZOS4
from json import load as json_load
from json import dump as json_dump
from pathlib import Path
from typing import Dict
from typing import List


class Config:
    """Central configuration."""

    # Camera settings
    IPHONE_IP: str = "169.254.124.190"
    IPHONE_PORT: int = 8080

    # Capture settings
    CAPTURE_INTERVAL: int = 20  # seconds

    # Paths
    ROOT_DIR: Path = Path(__file__).parent.parent.parent
    # Data Paths
    DATA_DIR: Path = ROOT_DIR / "data"
    ARUCO_DIR: Path = DATA_DIR / "arucos"
    CAPTURES_DIR: Path = DATA_DIR / "captures"
    TILES_DIR: Path = DATA_DIR / "tiles"
    RESULTS_DIR: Path = DATA_DIR / "results"
    # Dataset Paths
    DATASET_DIR: Path = ROOT_DIR / "dataset"
    YOLO_MODEL: Path = ROOT_DIR / "checkpoints" / "tile_detection_model_l.pt"
    YOLO_URL: str = "https://www.research-collection.ethz.ch/bitstreams/2fccd1fd-b97a-476e-87b7-48fd8a124c83/download"
    SAM2_MODEL: Path = ROOT_DIR / "checkpoints" / "sam2.1_hiera_small.pt"
    SAM2_URL: str = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"

    ARUCO_MARKER_SIZE_MM: int = 50
    ARUCO_PRINT_DPI: int = 300
    ARUCO_BORDER_BITS: int = 1

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create output directories if they don't exist."""
        cls.ARUCO_DIR.mkdir(parents=True, exist_ok=True)
        cls.CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        cls.TILES_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load_aruco_coords(cls, path: Path) -> None:
        """Load from JSON file."""
        with open(path) as f:
            data = json_load(f)
            cls.ARUCO_WORLD_COORDS = {int(k): v for k, v in data.items()}

    @classmethod
    def save_aruco_coords(cls, path: Path) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json_dump(cls.ARUCO_WORLD_COORDS, f, indent=2)

    @staticmethod
    def scale_image(image, scaling_factor):
        """
        Scale image. Uses INTER_AREA for downscaling and INTER_LANCZOS4 for upscaling.
        """
        if scaling_factor < 1:
            interpolation = INTER_AREA
        else:
            interpolation = INTER_LANCZOS4

        return cv2_resize(
            src=image,
            dsize=None,
            fx=scaling_factor,
            fy=scaling_factor,
            interpolation=interpolation,
        )
