"""
Build the publishable copy of the tile database.

Reads the working database and writes the publishable copy: full-frame masks are
replaced by tight per-tile crops (plus a 1 px margin), the original frame position is
kept as a bounding box, and provenance attributes are added. Prints SHA-256 checksums
for the deposit record.

    uv run prepare_release.py [--src data/tiles/tiles_db.h5]
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy

HERE = Path(__file__).parent
META = json.loads((HERE / "metadata.json").read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src", type=Path, default=HERE / "data" / "tiles" / "tiles_db.h5"
    )
    parser.add_argument("--out", type=Path, default=HERE / "tile_db.h5")
    parser.add_argument(
        "--margin", type=int, default=1, help="blank pixels around each crop"
    )
    args = parser.parse_args()

    shutil.copy2(args.src.parent / "tile_metrics.csv", HERE / "tile_metrics.csv")

    with h5py.File(args.src, "r") as src, h5py.File(args.out, "w") as out:
        masks = src["masks"]
        n, height, width = masks.shape

        for name in src:
            if name != "masks":
                src.copy(name, out)

        group = out.create_group("masks")
        bboxes = numpy.zeros((n, 4), dtype=numpy.int32)

        for i in range(n):
            ys, xs = numpy.where(masks[i])
            y0 = max(ys.min() - args.margin, 0)
            y1 = min(ys.max() + 1 + args.margin, height)
            x0 = max(xs.min() - args.margin, 0)
            x1 = min(xs.max() + 1 + args.margin, width)
            bboxes[i] = (y0, y1, x0, x1)

            group.create_dataset(
                f"{i:05d}",
                data=masks[i, y0:y1, x0:x1],
                compression="gzip",
                compression_opts=4,
            )
            if i % 250 == 0:
                print(f"  cropping {i}/{n}", flush=True)

        out.create_dataset("bboxes", data=bboxes, compression="gzip")

        out.attrs["title"] = META["titles"][0]["title"]
        out.attrs["version"] = META["version"]
        out.attrs["license"] = META["rightsList"][0]["rightsIdentifier"]
        out.attrs["pixel_scale_units"] = "mm/px"

        # DOIs are minted late; an unfilled placeholder is simply left out.
        for attr, value in (
            ("doi", META["identifiers"][0]["identifier"]),
            ("paper_doi", META["relatedIdentifiers"][0]["relatedIdentifier"]),
        ):
            if not value.startswith("TODO"):
                out.attrs[attr] = value

        out.attrs["crop_margin_px"] = args.margin
        out.attrs["frame_shape"] = (height, width)
        out.attrs["n_tiles"] = n

    # The crop must not have clipped anything: pixel counts still match the stored areas.
    with h5py.File(args.out, "r") as f:
        areas = f["area_px"][:]
        for i in (0, len(areas) // 2, len(areas) - 1):
            assert f[f"masks/{i:05d}"][:].sum() == areas[i], f"tile {i} lost pixels"

    for path in (args.out, HERE / "tile_metrics.csv"):
        print(f"{sha256(path)}  {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
