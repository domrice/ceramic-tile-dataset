"""
Plot every analysis figure for the mosaic tile dataset.

All metrics are stored in the HDF5 file, so nothing is recomputed here.

    uv run plot_tiles.py [--db tile_db.h5] [--out figures]

Set TILE_PLOT_FONT to an .otf file to match the typeface used in the paper figures.
"""

import argparse
import os
from pathlib import Path

import h5py
import matplotlib
import numpy

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# The 12 physical tile colours; true_colors holds exactly these RGB values.
PALETTE = {
    "yellow": (191, 148, 75),
    "orange": (189, 107, 65),
    "purple": (65, 65, 85),
    "dark_blue": (44, 58, 96),
    "light_blue": (59, 104, 121),
    "green": (109, 142, 109),
    "red": (122, 54, 45),
    "brown": (72, 60, 50),
    "white": (180, 180, 180),
    "dark_gray": (90, 90, 90),
    "light_gray": (140, 140, 140),
    "black": (20, 20, 12),
}

METRICS = (
    "area_px",
    "area_mm2",
    "eccentricity",
    "solidity",
    "aspect_ratio",
    "compactness",
    "circularity",
)

# Colourblind-safe accents for the histogram.
ACCENTS = ["#0173b2", "#de8f05", "#949494"]

# (dataset, axis label, y limits) for the area scatter plots.
AREA_PLOTS = [
    ("circularity", "Circularity (1=circle)", (0, 1)),
    ("solidity", "Solidity (1=convex)", (0, 1)),
    ("eccentricity", "Eccentricity (0=circle)", (0, 1)),
    ("aspect_ratio", "Aspect Ratio", None),
    ("compactness", "Compactness (area / perimeter²)", None),
]


def setup_style() -> None:
    font_path = os.environ.get("TILE_PLOT_FONT")
    if font_path and Path(font_path).is_file():
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        # Only override mathtext for a custom face; the default fontset is complete.
        plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": font_name})
    else:
        font_name = "DejaVu Sans"  # always shipped with matplotlib

    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "lines.markersize": 4.8,
        }
    )


def load(db_path: Path) -> dict:
    """Read every stored array, dropping degenerate tiles."""
    with h5py.File(db_path, "r") as f:
        n = len(f["masks"])
        data = {k: f[k][:] for k in METRICS}
        data["colors"] = f["colors"][:]
        data["true_colors"] = f["true_colors"][:]

    assert all(len(v) == n for v in data.values()), "dataset lengths disagree"
    palette_rgb = set(PALETTE.values())
    assert all(
        tuple(c) in palette_rgb for c in numpy.unique(data["true_colors"], axis=0)
    ), "true_colors holds values outside the palette"

    keep = data["area_mm2"] > 0
    return {k: v[keep] for k, v in data.items()}


def color_counts(true_colors: numpy.ndarray):
    """Tile count per palette colour, sorted by frequency."""
    names = numpy.array(list(PALETTE))
    rgb = numpy.array(list(PALETTE.values()))
    counts = numpy.array([(true_colors == c).all(axis=1).sum() for c in rgb])

    order = numpy.argsort(counts)[::-1]
    return names[order], counts[order], rgb[order]


def scatter(x, y, colors, xlabel, ylabel, out, ylim=None):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.scatter(x, y, c=colors)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="both", linestyle="solid", alpha=0.5)
    if ylim:
        ax.set_ylim(*ylim)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_histogram(area, out):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.hist(area, bins="fd", color=ACCENTS[2], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Area (mm²)")
    ax.set_ylabel("Tile Quantity")

    for value, color, label in (
        (area.mean(), ACCENTS[0], "Mean"),
        (numpy.median(area), ACCENTS[1], "Median"),
    ):
        ax.axvline(
            value,
            color=color,
            linestyle="dashed",
            linewidth=2.0,
            label=f"{label}: {value:.1f} mm²",
        )

    ax.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncols=4,
        borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_correlation(data, out):
    cols = ["area_mm2", "eccentricity", "solidity", "aspect_ratio", "compactness", "circularity"]
    corr = numpy.corrcoef(numpy.array([data[c] for c in cols]))

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(
                j,
                i,
                f"{corr[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=16,
                color="white" if abs(corr[i, j]) > 0.6 else "black",
            )

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_color_distribution(names, counts, rgb, total, out):
    fig, ax = plt.subplots(figsize=(12, 6))
    y = numpy.arange(len(counts))
    ax.barh(y, counts, color=rgb / 255.0, edgecolor="black", linewidth=1.5)

    for i, (count, color) in enumerate(zip(counts, rgb)):
        if count == 0:
            continue
        ax.text(
            count + total * 0.01,
            i,
            f"{count} ({count / total * 100:.1f}%)",
            va="center",
            fontsize=10,
            color="black",
        )
        ax.text(
            count / 2,
            i,
            f"RGB({color[0]}, {color[1]}, {color[2]})",
            va="center",
            ha="center",
            fontsize=8,
            color="white" if color.mean() < 128 else "black",
            weight="bold",
        )

    ax.set_yticks(y, names)
    ax.set_xlabel("Number of Tiles")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_color_space(colors, true_colors, out):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        colors[:, 0],
        colors[:, 1],
        colors[:, 2],
        c=true_colors / 255.0,
        s=20,
        alpha=0.6,
        edgecolors="gray",
        linewidth=0.3,
    )
    ax.set_xlabel("Red", fontsize=12)
    ax.set_ylabel("Green", fontsize=12)
    ax.set_zlabel("Blue", fontsize=12)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_palette(counts, rgb, total, out):
    fig, ax = plt.subplots(figsize=(12, 2))

    for i, (count, color) in enumerate(zip(counts, rgb)):
        ax.add_patch(
            plt.Rectangle((i, 0), 1, 1, facecolor=color / 255.0, edgecolor="black", linewidth=2)
        )
        if count == 0:
            continue
        ax.text(
            i + 0.5,
            0.5,
            f"{count / total * 100:.1f}%",
            ha="center",
            va="center",
            fontsize=16,
            color="white" if color.mean() < 128 else "black",
            weight="bold",
        )

    ax.set_xlim(0, len(rgb))
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=here / "tile_db.h5")
    parser.add_argument("--out", type=Path, default=here / "figures")
    args = parser.parse_args()

    data = load(args.db)
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()

    rgb_points = data["colors"] / 255.0
    area = data["area_mm2"]
    total = len(area)

    scatter(
        data["eccentricity"],
        data["solidity"],
        rgb_points,
        "Eccentricity (0=circle)",
        "Solidity (1=convex)",
        args.out / "eccentricity_vs_solidity.pdf",
        ylim=(0, 1),
    )
    for metric, label, ylim in AREA_PLOTS:
        scatter(
            area,
            data[metric],
            rgb_points,
            "Area (mm²)",
            label,
            args.out / f"area_vs_{metric}.pdf",
            ylim=ylim,
        )

    plot_histogram(area, args.out / "area_histogram.pdf")
    plot_correlation(data, args.out / "correlation_heatmap.pdf")

    names, counts, rgb = color_counts(data["true_colors"])
    plot_color_distribution(names, counts, rgb, total, args.out / "color_distribution.pdf")
    plot_color_space(data["colors"], data["true_colors"], args.out / "color_space_3d.pdf")
    plot_palette(counts, rgb, total, args.out / "color_palette.pdf")

    print(f"{total} tiles")
    print(f"  area (mm²): mean {area.mean():.2f}  median {numpy.median(area):.2f}  "
          f"min {area.min():.2f}  max {area.max():.2f}")
    for metric in ("eccentricity", "solidity", "circularity"):
        print(f"  {metric}: mean {data[metric].mean():.3f}")
    for name, count in zip(names, counts):
        print(f"  {name}: {count} ({count / total * 100:.1f}%)")
    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()
