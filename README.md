# Ceramic Tile Dataset

[![DOI](https://img.shields.io/badge/DOI-10.3929%2Fethz--c--000806582-215CAF)](https://doi.org/10.3929/ethz-c-000806582)
[![Data license: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Format: HDF5](https://img.shields.io/badge/format-HDF5-orange)](https://www.hdfgroup.org/solutions/hdf5/)
[![Code license](https://img.shields.io/github/license/domrice/ceramic-tile-dataset?label=code)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Binary masks, colours and shape metrics for **3301 physical ceramic tiles**, photographed on a
calibrated board, scaled to millimetres via ArUco fiducials, and segmented with YOLO + SAM2.

## Quick start

```python
import h5py

with h5py.File("tile_db.h5", "r") as f:
    mask = f["masks/00042"][:]   # bool array, cropped to the tile
    color = f["colors"][42]      # mean RGB
    area = f["area_mm2"][42]     # mm²
```

Any HDF5 reader works — no code from this repository is required.

## Contents

| File                 | Content                                                         |
| -------------------- | --------------------------------------------------------------- |
| `tile_db.h5`         | The dataset (HDF5, gzip-compressed)                             |
| `tile_metrics.csv`   | Per-tile metrics as a flat table, without masks                 |
| `metadata.json`      | DataCite record and full HDF5 schema                            |
| `plot_tiles.py`      | Regenerates all analysis figures                                |
| `prepare_release.py` | Builds `tile_db.h5` from the working database in `data/tiles/`  |
| `dataset/`           | Digitization pipeline: capture, ArUco scaling, YOLO + SAM2      |
| `ios_app/`           | iPhone camera server the pipeline captures from                 |

## Schema

`N = 3301`; every array is indexed by tile id `0 … N-1`.

| Dataset           | Shape    | Type    | Meaning                                                                 |
| ----------------- | -------- | ------- | ----------------------------------------------------------------------- |
| `masks/00000` …   | varies   | bool    | Tight binary mask per tile (+1 px margin), named by zero-padded id      |
| `bboxes`          | `(N, 4)` | int32   | Crop position in the 4032×3024 frame, `[y0, y1, x0, x1]`, end-exclusive |
| `colors`          | `(N, 3)` | uint8   | Mean RGB of the tile surface                                            |
| `true_colors`     | `(N, 3)` | uint8   | Colour quantised to the 12-colour physical palette                      |
| `scaling_factors` | `(N,)`   | float32 | Scale in mm/px                                                          |
| `timestamps`      | `(N,)`   | string  | Capture time, `YYYYMMDD_HHMMSS`                                         |
| `area_px`         | `(N,)`   | float32 | Mask area in pixels                                                     |
| `area_mm2`        | `(N,)`   | float32 | `area_px · scaling_factor²`                                             |
| `eccentricity`    | `(N,)`   | float32 | Of the equivalent ellipse; 0 = circle                                   |
| `solidity`        | `(N,)`   | float32 | Area / convex-hull area; 1 = convex                                     |
| `aspect_ratio`    | `(N,)`   | float32 | Major / minor axis of the equivalent ellipse                            |
| `compactness`     | `(N,)`   | float32 | `area / perimeter²`                                                     |
| `circularity`     | `(N,)`   | float32 | `4π · compactness`, clipped to `[0, 1]`; 1 = circle                     |

Shape descriptors follow [`skimage.measure.regionprops`](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops), computed on the full-resolution mask.

## Figures

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run plot_tiles.py   # writes figures/*.pdf and prints summary statistics
```

The script only reads stored metrics; nothing is recomputed. Set `TILE_PLOT_FONT` to an `.otf`
path to match the published typeface.

## Digitization pipeline

Recreates the database from physical tiles — not needed to use the data.

```bash
uv sync --group capture                           # torch, ultralytics, SAM2
uv run --group capture python generate_arucos.py  # printable ArUco sheets -> data/arucos/
uv run --group capture python -m dataset          # capture loop -> data/tiles/tiles_db.h5
uv run prepare_release.py                         # crop and package -> tile_db.h5
```

- **Camera:** `ios_app/` (Xcode) serves images over HTTP; set its address in `dataset/utils/config.py`.
- **Weights** download to `checkpoints/` on first use: SAM2 (`sam2.1_hiera_small.pt`, ~180 MB) and
  the YOLO tile detector (`tile_detection_model_l.pt`, 109 MB, from the
  [data deposit](https://doi.org/10.3929/ethz-c-000806582)). Retrain with `dataset/yolo/`.
- `uv run --group capture export_tray.py <photo> <tray.json>` runs detection on a single photo.

## License & citation

The data is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), the code under the [MIT License](LICENSE).
Please cite both the [dataset](https://doi.org/10.3929/ethz-c-000806582) and the related
publication; DOIs are in `metadata.json`.
