from dataset.camera.aruco import ArucoDetector
from dataset.utils.config import Config

Config.ensure_dirs()

ad = ArucoDetector()

ad.save_markers(
    num_markers=9,
    output_dir=Config.ARUCO_DIR,
    marker_size_mm=Config.ARUCO_MARKER_SIZE_MM,
    dpi=Config.ARUCO_PRINT_DPI,
    border_bits=Config.ARUCO_BORDER_BITS,
)
