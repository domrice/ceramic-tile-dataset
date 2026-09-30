"""
Camera capture and calibration module.
Handles iPhone HTTP capture and ArUco detection.
"""

import os

HERE = os.path.dirname(__file__)

from .capture import CameraCapture
from .aruco import ArucoDetector

__all__ = ["CameraCapture", "ArucoDetector"]
