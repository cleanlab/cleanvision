import math
from typing import List, Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
from PIL import Image


class VizManager:
    @staticmethod
    def individual_images(
        filepaths: List[str], titles: List[str], ncols: int, cell_size: Tuple[int, int]
    ) -> None:
        plot_image_grid(filepaths, titles, ncols, cell_size)

    @staticmethod
    def image_sets(
        filepath_sets: List[List[str]],
        title_sets: List[List[str]],
        ncols: int,
        cell_size: Tuple[int, int],
    ) -> None:
        for i, s in enumerate(filepath_sets):
            print(f"Set: {i}")
            plot_image_grid(s, title_sets[i], ncols, cell_size)


def set_image_on_axes(path: str, ax: matplotlib.axes.Axes, title: str) -> None:
    image = Image.open(path)
    cmap = "gray" if image.mode == "L" else None
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(title, fontsize=7)
    ax.imshow(image, cmap=cmap, vmin=0, vmax=255)


def plot_image_grid(
    filepaths: List[str], titles: List[str], ncols: int, cell_size: Tuple[int, int]
) -> None:
    nrows = math.ceil(len(filepaths) / ncols)
    ncols = min(ncols, len(filepaths))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(cell_size[0] * ncols, cell_size[1] * nrows)
    )
    if nrows > 1:
        idx = 0
        for i in range(nrows):
            for j in range(ncols):
                idx = i * ncols + j
                if idx >= len(filepaths):
                    axes[i, j].axis("off")
                    continue
                set_image_on_axes(filepaths[idx], axes[i, j], titles[idx])
            if idx >= len(filepaths):
                axes[i, j].axis("off")
    elif ncols > 1:
        for i in range(ncols):
            if i < len(filepaths):
                set_image_on_axes(filepaths[i], axes[i], titles[i])
    else:
        set_image_on_axes(filepaths[0], axes, titles[0])
    plt.show()
