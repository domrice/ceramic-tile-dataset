import numpy
import h5py
import pandas
import colorsys
from tqdm import trange
from skimage.measure import regionprops
from pathlib import Path
from numpy.typing import NDArray
from typing import Tuple
from typing import List
from typing import Dict
from typing import Optional
from datetime import datetime


class TileDatabase:
    """
    HDF5-based tile storage with immediate append.

    File structure:
        tiles.h5
        ├── masks [N x H x W] uint8 - Binary masks
        ├── colors [N x 3] uint8 - RGB colors
        ├── timestamps [N] string - Capture timestamps
        └── metadata (attributes) - Image dimensions, scaling factors, etc.
    """

    def __init__(self, db_path: Path, mask_shape: Optional[Tuple[int, int]]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.h5file = h5py.File(self.db_path, "a")

        if mask_shape is not None:

            self.h5file.create_dataset(
                "masks",
                shape=(0, *mask_shape),
                maxshape=(None, *mask_shape),
                dtype="bool",
                compression="gzip",
                compression_opts=4,
                chunks=(1, *mask_shape),
            )

            self.h5file.create_dataset(
                "colors",
                shape=(0, 3),
                maxshape=(None, 3),
                dtype="uint8",
                chunks=(1000, 3),
            )

            self.h5file.create_dataset(
                "scaling_factors",
                shape=(0,),
                maxshape=(None,),
                dtype="float32",
                chunks=(1000,),
            )

            dt = h5py.string_dtype(encoding="utf-8")
            self.h5file.create_dataset(
                "timestamps", shape=(0,), maxshape=(None,), dtype=dt, chunks=(1000,)
            )

            self.h5file.attrs["mask_height"] = mask_shape[0]
            self.h5file.attrs["mask_width"] = mask_shape[1]
            self.h5file.attrs["created"] = datetime.now().isoformat()
        else:
            print(f"Opened existing database: {self.db_path}")
            print(f"Current tiles: {len(self.h5file['masks'])}")

    def __len__(self) -> int:
        return len(self.h5file["masks"])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.h5file.close()
        print(f"Closed database: {self.db_path}")

    def add_tile(
        self,
        mask: NDArray,
        color: Tuple[int, int, int],
        scaling_factor: numpy.float32,
        timestamp: Optional[str],
    ) -> int:
        expected_shape = (
            self.h5file.attrs["mask_height"],
            self.h5file.attrs["mask_width"],
        )

        if mask.shape != expected_shape:
            raise ValueError(
                f"Mask shape {mask.shape}, does not match database {expected_shape}"
            )

        current_count = self.h5file["masks"].shape[0]
        tile_id = current_count

        self.h5file["masks"].resize((current_count + 1, *expected_shape))
        self.h5file["colors"].resize((current_count + 1, 3))
        self.h5file["scaling_factors"].resize((current_count + 1,))
        self.h5file["timestamps"].resize((current_count + 1,))

        self.h5file["masks"][tile_id] = mask
        self.h5file["colors"][tile_id] = color
        self.h5file["scaling_factors"][tile_id] = scaling_factor
        self.h5file["timestamps"][tile_id] = timestamp

        self.h5file.flush()

        return tile_id

    def add_tiles_batch(
        self,
        masks: List[NDArray],
        colors: List[Tuple[int, int, int]],
        scaling_factor: numpy.float32,
        timestamp: str,
    ) -> List[int]:
        if len(masks) != len(colors):
            raise ValueError("masks and colors must have the same length")

        expected_shape = (
            self.h5file.attrs["mask_height"],
            self.h5file.attrs["mask_width"],
        )

        for i, mask in enumerate(masks):
            if mask.shape != expected_shape:
                raise ValueError(
                    f"Mask {i} shape {mask.shape}, does not match database {expected_shape}"
                )

        current_count = self.h5file["masks"].shape[0]
        new_count = current_count + len(masks)

        self.h5file["masks"].resize((new_count, *expected_shape))
        self.h5file["colors"].resize((new_count, 3))
        self.h5file["scaling_factors"].resize((new_count,))
        self.h5file["timestamps"].resize((new_count,))

        for i, (mask, color) in enumerate(zip(masks, colors)):
            tile_id = current_count + i
            self.h5file["masks"][tile_id] = mask
            self.h5file["colors"][tile_id] = color
            self.h5file["scaling_factors"][tile_id] = scaling_factor
            self.h5file["timestamps"][tile_id] = timestamp

        self.h5file.flush()

        tile_ids = list(range(current_count, new_count))
        print(f"Saved {len(masks)} tiles with IDs {current_count}–{new_count-1}")

        return tile_ids

    def get_tile(self, tile_id: int) -> Dict:
        if tile_id >= len(self.h5file["masks"]):
            raise IndexError(f"Tile {tile_id} not found (max: {len(self) - 1})")

        result = {
            "mask": self.h5file["masks"][tile_id],
            "color": self.h5file["colors"][tile_id],
            "scaling_factor": self.h5file["scaling_factors"][tile_id],
            "area": self.h5file["area_mm2"][tile_id],
            "timestamp": self.h5file["timestamps"][tile_id],
        }

        if "true_colors" in self.h5file:
            result["true_color"] = tuple(
                int(c) for c in self.h5file["true_colors"][tile_id]
            )

        return result

    def get_all_tiles(self) -> List[Dict]:
        num_tiles = len(self)

        if num_tiles == 0:
            return []

        masks = self.h5file["masks"][:]
        colors = self.h5file["colors"][:]
        scaling_factors = self.h5file["scaling_factors"][:]
        area = self.h5file["area_mm2"][:]
        timestamps = self.h5file["timestamps"][:]

        tiles = []
        for i, (mask, color, sf, a, ts) in enumerate(
            zip(masks, colors, scaling_factors, area, timestamps)
        ):
            tiles.append(
                {
                    "tile_id": i,
                    "mask": mask,
                    "color": color,
                    "scaling_factor": sf,
                    "area": a,
                    "timestamp": ts,
                }
            )

        return tiles

    def get_tiles_range(self, start: int, end: int) -> List[Dict]:
        if start < 0 or end > len(self):
            raise IndexError(f"Range [{start}, {end}] out of bounds [0, {len(self)}]")

        if start >= end:
            return []

        masks = self.h5file["masks"][start:end]
        colors = self.h5file["colors"][start:end]
        scaling_factors = self.h5file["scaling_factors"][start:end]
        area = self.h5file["area_mm2"][start:end]
        timestamps = self.h5file["timestamps"][start:end]

        tiles = []
        for i, (mask, color, sf, a, ts) in enumerate(
            zip(masks, colors, scaling_factors, area, timestamps)
        ):
            tiles.append(
                {
                    "tile_id": i,
                    "mask": mask,
                    "color": color,
                    "scaling_factor": sf,
                    "area": a,
                    "timestamp": ts,
                }
            )

        return tiles

    def get_tiles_by_metric(
        self, metric: str, min_val: float, max_val: float
    ) -> List[int]:
        if metric not in self.h5file:
            raise ValueError(
                f"Metric '{metric}' not found in dataset. Compute metrics first."
            )

        values = self.h5file[metric][:]
        ids = numpy.where((values >= min_val) & (values <= max_val))[0]
        return ids.tolist()

    def get_tiles_by_color(
        self, target_color: Tuple[int, int, int], tolerance: int = 30
    ) -> List[int]:
        colors = self.h5file["colors"][:]

        target = numpy.array(target_color)
        distance = numpy.sqrt(numpy.sum((colors - target) ** 2, axis=1))
        ids = numpy.where(distance <= tolerance)[0]

        return ids

    def get_all_colors(self) -> NDArray:
        return self.h5file["colors"][:]

    def get_stats(self) -> Dict:
        colors = self.h5file["colors"][:]

        return {
            "num_tiles": len(self),
            "mask_shape": (
                self.h5file.attrs["mask_height"],
                self.h5file.attrs["mask_width"],
            ),
            "mask_dtype": str(self.h5file["masks"].dtype),
            "file_size_mb": self.db_path.stat().st_size / 1024 / 1024,
            "compression": {
                "algorithm": self.h5file["masks"].compression,
                "level": self.h5file["masks"].compression_opts,
            },
            "color_distribution": {
                "mean_rgb": tuple(numpy.mean(colors, axis=0).astype(int)),
                "std_rgb": tuple(numpy.std(colors, axis=0).astype(int)),
            },
            "created": self.h5file.attrs.get("created", "unknown"),
            "timestamps": {
                "first": self.h5file["timestamps"][0],
                "last": self.h5file["timestamps"][-1],
            },
        }

    def apply_true_palette(
        self, true_palette: Dict[str, Tuple[int, int, int]], overwrite: bool = False
    ) -> None:
        colors = self.h5file["colors"][:]
        labels, _ = TileDatabase.map_to_true_colors(colors, true_palette)
        palette_array = numpy.array(list(true_palette.values()), dtype=numpy.uint8)
        mapped = palette_array[labels]

        if "true_colors" in self.h5file:
            if overwrite:
                del self.h5file["true_colors"]
            else:
                print("True colors already part of database!")
                return

        self.h5file.create_dataset(
            "true_colors", data=mapped, dtype="uint8", compression="gzip"
        )
        self.h5file.flush()

    @staticmethod
    def map_to_true_colors(
        colors_array: NDArray,
        true_palette: Dict[str, Tuple[int, int, int]],
        sat_thresh: float = 0.15,
    ) -> Tuple[NDArray, List[str]]:
        palette_names = list(true_palette.keys())
        palette_colors = numpy.array(list(true_palette.values()))

        def rgb_to_hsv_array(rgb_array):
            hsv = numpy.zeros_like(rgb_array, dtype=numpy.float32)
            for i in range(len(rgb_array)):
                r, g, b = rgb_array[i] / 255.0
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                hsv[i] = [h * 360, s, v]
            return hsv

        tiles_hsv = rgb_to_hsv_array(colors_array)
        palette_hsv = rgb_to_hsv_array(palette_colors)

        palette_is_achromatic = palette_hsv[:, 1] < sat_thresh

        achromatic_indices = numpy.where(palette_is_achromatic)[0]
        chromatic_indices = numpy.where(~palette_is_achromatic)[0]

        color_labels = numpy.zeros(len(colors_array), dtype=int)

        for i, (tile_rgb, tile_hsv) in enumerate(zip(colors_array, tiles_hsv)):
            h_tile, s_tile, v_tile = tile_hsv

            if s_tile < sat_thresh:
                if len(achromatic_indices) == 0:
                    distances = numpy.sqrt(
                        numpy.sum((palette_colors - tile_rgb) ** 2, axis=1)
                    )
                    color_labels[i] = numpy.argmin(distances)
                else:
                    achromatic_values = palette_hsv[achromatic_indices, 2]
                    value_distance = numpy.abs(achromatic_values - v_tile)
                    best_achromatic = achromatic_indices[numpy.argmin(value_distance)]
                    color_labels[i] = best_achromatic
            else:
                if len(chromatic_indices) == 0:
                    distances = numpy.sqrt(
                        numpy.sum((palette_colors - tile_rgb) ** 2, axis=1)
                    )
                    color_labels[i] = numpy.argmin(distances)
                else:
                    chromatic_hues = palette_hsv[chromatic_indices, 0]

                    hue_diff = numpy.abs(chromatic_hues - h_tile)
                    hue_diff = numpy.minimum(hue_diff, 360 - hue_diff)

                    chromatic_saturations = palette_hsv[chromatic_indices, 1]
                    chromatic_values = palette_hsv[chromatic_indices, 2]

                    sat_diff = numpy.abs(chromatic_saturations - s_tile)
                    val_diff = numpy.abs(chromatic_values - v_tile)

                    combined_distance = (
                        hue_diff / 180.0 + sat_diff * 0.3 + val_diff * 0.2
                    )

                    best_chromatic = chromatic_indices[numpy.argmin(combined_distance)]
                    color_labels[i] = best_chromatic

        return color_labels, palette_names

    @staticmethod
    def compute_tile_geometry_metrics(mask: NDArray) -> Dict:
        labeled_mask = mask.astype(numpy.uint8)
        props = regionprops(labeled_mask)[0]

        area = numpy.float32(props.area)
        perimeter = props.perimeter
        major_axis = props.axis_major_length
        minor_axis = props.axis_minor_length

        if minor_axis > 0:
            aspect_ratio = major_axis / minor_axis
        else:
            aspect_ratio = 0.0

        if perimeter > 0:
            compactness = area / (perimeter**2)
            circularity = 4 * numpy.pi * compactness
        else:
            compactness = 0.0
            circularity = 0.0

        return {
            "area_px": area,
            "eccentricity": props.eccentricity,
            "solidity": props.solidity,
            "aspect_ratio": numpy.float32(aspect_ratio),
            "compactness": numpy.float32(compactness),
            "circularity": numpy.clip(numpy.float32(circularity), 0, 1),
        }

    def store_metrics_in_hdf5(self, metrics_df: pandas.DataFrame) -> None:
        num_tiles = len(metrics_df)

        for column in [
            "area_px",
            "area_mm2",
            "eccentricity",
            "solidity",
            "aspect_ratio",
            "compactness",
            "circularity",
        ]:
            if column in self.h5file:
                del self.h5file[column]

            self.h5file.create_dataset(
                column,
                data=metrics_df[column].values,
                dtype="float32",
                compression="gzip",
            )

        self.h5file.flush()
        print(f"Stored metrics for {num_tiles} tiles in HDF5 file!")

    def compute_all_metrics(
        self, save_to: Optional[Path] = None, store_in_hdf5: bool = True
    ) -> pandas.DataFrame:
        """
        Compute shape metrics for all tiles in database.
        """
        records = []

        for tile_id in trange(len(self), desc="Metrics", unit=" Tiles", colour="blue"):
            tile = self.get_tile(tile_id)
            metrics = TileDatabase.compute_tile_geometry_metrics(tile["mask"])

            if metrics is None:
                continue

            record = {
                "tile_id": tile_id,
                "color": tile["color"],
                "scaling_factor": tile["scaling_factor"],
                "timestamp": tile["timestamp"],
                "area_mm2": metrics["area_px"] * (tile["scaling_factor"] ** 2),
                **metrics,
            }

            records.append(record)

        df = pandas.DataFrame(records)

        if save_to:
            df.to_csv(save_to, index=False)

        if store_in_hdf5:
            self.store_metrics_in_hdf5(df)

        return df
