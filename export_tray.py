"""Run the real YOLO + SAM2 detection on one capture photo and write the boxes,
outlines and colours as JSON, in photo pixel coordinates.

This is part A of the paper's animation data; the packing repository's
`scripts/_export_animation.py` reads the resulting `tray.json`.

    uv run --group capture export_tray.py <capture.jpg> <tray.json>
"""

import json
import sys

from pathlib import Path

import cv2
import numpy as np

from skimage.measure import approximate_polygon
from skimage.measure import find_contours

from dataset.segmentation import Segmentation
from dataset.utils.config import Config

YOLO_CONF = 0.7
POLYGON_TOLERANCE = 2.0  # pixels; the outline is drawn at ~1/2 scale in the video


def mask_polygon(mask, tolerance=POLYGON_TOLERANCE):
    """The largest contour of a mask as an [[x, y], ...] pixel polygon."""
    contours = find_contours(mask > 0, 0.5)
    if not contours:
        return None
    contour = max(contours, key=len)
    poly = approximate_polygon(contour, tolerance=tolerance)
    return np.fliplr(poly).round(1).tolist()  # (row, col) -> (x, y)


def export_tray(image_path, out_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"No capture photo at {image_path}")

    segmentation = Segmentation(
        yolo_model_path=Config.YOLO_MODEL,
        sam2_model_path=Config.SAM2_MODEL,
        yolo_conf_threshold=YOLO_CONF,
    )
    segmentation.set_image(image)

    tiles = []
    for tile in segmentation.segment_by_points():
        polygon = mask_polygon(tile.mask)
        if polygon is None:
            continue
        r, g, b = tile.color  # _extract_color already returns RGB
        tiles.append(
            {
                "id": len(tiles),
                "bbox": list(tile.bbox),
                "center": [int(c) for c in tile.center_px],
                "polygon": polygon,
                "color": [int(r), int(g), int(b)],
            }
        )

    height, width = image.shape[:2]
    out_path.write_text(
        json.dumps({"image_size": [width, height], "tiles": tiles}, indent=1)
    )
    print(f"[info] Exported: {out_path} ({len(tiles)} tiles)")


if __name__ == "__main__":
    export_tray(Path(sys.argv[1]), Path(sys.argv[2]))
