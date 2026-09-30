"""
iPhone camera HTTP capture client.
"""

import requests
import base64
import numpy as np
import cv2
from datetime import datetime
from typing import Optional
from typing import Tuple


class Colors:
    """
    ANSI color codes for terminal output
    """

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
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


class CameraCapture:
    """Captures images from iPhone camera server."""

    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.base_url = f"http://{ip}:{port}"

    @property
    def timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def health_check(self) -> bool:
        """Check if server is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def capture_test_image(self, timeout: bool = True) -> None:
        response = requests.post(f"{self.base_url}/capture")

        if response.status_code == 200 and len(response.content) > 0:
            data = response.json()

            # Decode image
            image_data = base64.b64decode(data["image_base64"])
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Show image
            if timeout:
                cv2.imshow("Test Image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return img

    def capture_image(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        print_status(f"  Capturing photo at {self.timestamp}", "info")
        response = requests.post(f"{self.base_url}/capture")

        if response.status_code == 200 and len(response.content) > 0:
            data = response.json()

            # Decode image
            image_data = base64.b64decode(data["image_base64"])
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Extract intrinsics
            K = np.array(data["camera_calibration"]["K"])
            dims = data["camera_calibration"]["image_dimensions"]

            distortion_center_data = data["camera_calibration"].get(
                "lens_distortion_center"
            )
            if distortion_center_data:
                distortion_center = (
                    distortion_center_data["x"],
                    distortion_center_data["y"],
                )
                print_status(
                    f"     Distortion center: ({distortion_center[0]:.1f}, {distortion_center[1]:.1f})",
                    "info",
                )
                print_status(
                    f"     Principal point: ({K[0,2]:.1f}, {K[1,2]:.1f})", "info"
                )

                offset_x = distortion_center[0] - K[0, 2]
                offset_y = distortion_center[1] - K[1, 2]
                offset_magnitude = np.sqrt(offset_x**2 + offset_y**2)
                if offset_magnitude < 50:
                    print_status(
                        f"     Offset: {offset_magnitude:.2f} pixels", "success"
                    )
                elif offset_magnitude >= 50 and offset_magnitude < 150:
                    print_status(
                        f"     Offset: {offset_magnitude:.2f} pixels", "warning"
                    )
                else:
                    print_status(f"     Offset: {offset_magnitude:.2f} pixels", "error")

            else:
                distortion_center = None

            lut = data["camera_calibration"].get("lens_distortion_lookup_table", [])
            dist_coeffs = CameraCapture._convert_apple_distortion_to_opencv(
                lut, dims["width"], dims["height"], K, distortion_center
            )

            print_status(
                f"     Image captured: {dims['width']}x{dims['height']}", "success"
            )
            print_status(
                f"     Focal length: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}", "success"
            )
            print_status(
                f"     Principal point: cx={K[0,2]:.1f}, cy={K[1,2]:.1f}", "success"
            )
            print_status(
                f"     Distortion coeffs (k1, k2, p1, p2, k3): {dist_coeffs}", "success"
            )

            print_status(f"Image size: {dims['width']} x {dims['height']}", "info")
            print_status(
                f"cx/width ratio: {K[0,2] / dims['width']:.3f}"
            ), "info"  # Should be ~0.5
            print_status(
                f"cy/height ratio: {K[1,2] / dims['height']:.3f}", "info"
            )  # Should be ~0.5

            return (img, K, dist_coeffs)
        else:
            print_status(f"     Error: Status {response.status_code}", "error")
            return None

    @staticmethod
    def _convert_apple_distortion_to_opencv(
        lut: list,
        image_width: int,
        image_height: int,
        K: np.ndarray,
        distortion_center: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Convert Apple's radial distortion LUT to OpenCV distortion coefficients.

        Apple's LUT: radius magnification factors from center to corner
        - Each value = magnification of that radius
        - Assumes radial symmetry (no tangential distortion)

        OpenCV model: r_corrected = r * (1 + k1*r² + k2*r⁴ + k3*r⁶)

        Args:
            lut: Apple's distortion lookup table (list of floats)
            image_width: Image width in pixels
            image_height: Image height in pixels
            K: 3x3 intrinsic matrix
            distortion_center: (x, y) distortion center, or None to use principal point

        Returns:
            np.array([k1, k2, p1, p2, k3]) - OpenCV distortion coefficients
        """
        lut_array = np.array(lut)

        non_zero_indices = np.where(lut_array != 0)[0]
        if len(non_zero_indices) == 0:
            print("⚠️  All LUT values are zero - no distortion")
            return np.zeros(5)

        last_valid_idx = non_zero_indices[-1] + 1
        lut_trimmed = lut_array[:last_valid_idx]

        if distortion_center is not None:
            cx, cy = distortion_center
        else:
            cx = K[0, 2]
            cy = K[1, 2]

        max_r_pixels = np.sqrt((image_width - cx) ** 2 + (image_height - cy) ** 2)
        fx = K[0, 0]

        n_samples = len(lut_trimmed)
        radii_normalized = np.linspace(0, 1, n_samples)
        radii_pixels = radii_normalized * max_r_pixels

        valid_indices = radii_pixels > 1.0

        if np.sum(valid_indices) < 3:
            print("⚠️  Insufficient valid samples")
            return np.zeros(5)

        r_px = radii_pixels[valid_indices]
        offset = lut_trimmed[valid_indices]

        magnification = 1.0 + offset / r_px
        r_norm = r_px / fx
        distortion = magnification - 1.0

        A = np.column_stack([r_norm**2, r_norm**4, r_norm**6])

        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(A, distortion, rcond=None)
            k1, k2, k3 = coeffs[0], coeffs[1], coeffs[2]

            if len(residuals) > 0:
                rmse = np.sqrt(residuals[0] / len(distortion))
                print(f"Distortion fit RMSE: {rmse:.6f}")

            print(f"OpenCV dist_coeffs:")
            print(f"   k1 = {k1:.8f}")
            print(f"   k2 = {k2:.8f}")
            print(f"   k3 = {k3:.8f}")

        except np.linalg.LinAlgError:
            print("⚠️  Linear algebra error")
            k1, k2, k3 = 0, 0, 0

        return np.array([k1, k2, 0.0, 0.0, k3])
