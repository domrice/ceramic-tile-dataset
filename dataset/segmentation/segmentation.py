import numpy
import torch

from cv2 import cvtColor
from cv2 import COLOR_BGR2RGB
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO

from dataset.utils.config import Config

from dataclasses import dataclass
from urllib.request import urlretrieve
from pathlib import Path
from typing import List
from typing import Tuple
from typing import Optional


def _download(path: Path, url: str) -> None:
    """Fetch a model weight file on first use."""
    if not path.exists():
        print(f"Downloading {url} to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, path)


@dataclass
class SegmentedTile:
    """
    Container for segmented object data.
    """

    mask: numpy.ndarray
    color: Tuple
    center_px: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # YOLO xyxy
    conf: float


class Segmentation:
    """
    SAM2-based segmentation for tile detection.

    Optimized for:
    - Single image segmentation
    - ArUco-based scaling (mm/pixel)
    - Batch processing of multiple tiles
    """

    def __init__(
        self,
        yolo_model_path: Optional[Path] = None,
        sam2_model_path: Optional[Path] = None,
        yolo_conf_threshold: float = 0.5,
    ):
        self.yolo = Segmentation._load_yolo(yolo_model_path)
        self.sam2 = Segmentation._load_sam2(sam2_model_path)
        self.yolo_conf_threshold = yolo_conf_threshold

        self._image = None
        self._image_set = False

    @staticmethod
    def _load_yolo(model_path: Optional[Path] = None) -> YOLO:
        _download(model_path, Config.YOLO_URL)
        return YOLO(str(model_path))

    @staticmethod
    def _load_sam2(model_path: Optional[Path] = None) -> SAM2ImagePredictor:
        if torch.cuda.is_available():
            device = torch.device("cuda")

            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        _download(model_path, Config.SAM2_URL)
        print(f"Loading SAM2 on {device}")

        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        sam2_model = build_sam2(str(model_cfg), str(model_path), device=device)

        return SAM2ImagePredictor(sam2_model)

    def set_image(self, image: numpy.ndarray) -> None:
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB/BGR image, got shape {image.shape}")

        self._image = image

        image_rgb = cvtColor(image, COLOR_BGR2RGB)
        self.sam2.set_image(image_rgb)
        self._image_set = True

    def segment_by_points(self) -> List[SegmentedTile]:
        if not self._image_set:
            raise RuntimeError("Call set_image() first!")

        yolo_results = self.yolo(self._image)
        detection = yolo_results[0].boxes

        if len(detection) == 0:
            print("No tiles detected by YOLO.")
            return []

        segmented_tiles = []

        for i, det in enumerate(detection):
            conf = det.conf.item()

            if conf < self.yolo_conf_threshold:
                print(
                    f"Skipping tile {i}: confidence {conf:.2f} < {self.yolo_conf_threshold}"
                )
                continue

            xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2

            point = numpy.array([[center_x, center_y]])
            label = numpy.array([1])  # foreground

            # the box, not just the centre point: a point alone lets SAM2 latch onto
            # an interior feature (a specular highlight, a crack) instead of the tile
            masks, scores, _ = self.sam2.predict(
                point_coords=point,
                point_labels=label,
                box=xyxy[None, :],
                multimask_output=False,
            )

            if masks is None or len(masks) == 0:
                print(f"SAM2 failed for tile {i} at ({center_x}, {center_y})")
                continue

            mask = masks[0].astype(numpy.uint8) * 255
            color = self._extract_color(mask)

            segmented_tiles.append(
                SegmentedTile(
                    mask=mask,
                    color=color,
                    center_px=(center_x, center_y),
                    bbox=(int(xmin), int(ymin), int(xmax), int(ymax)),
                    conf=conf,
                )
            )

        return segmented_tiles

    def _extract_color(self, mask: numpy.ndarray) -> Tuple[int, int, int]:
        masked_pixels = self._image[mask > 0]

        if len(masked_pixels) == 0:
            return (0, 0, 0)

        mean_bgr = numpy.mean(masked_pixels, axis=0).astype(int)
        mean_rgb = (mean_bgr[2], mean_bgr[1], mean_bgr[0])

        return tuple(mean_rgb)
