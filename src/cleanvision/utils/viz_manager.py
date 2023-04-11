from typing import List, Tuple

import math
import matplotlib.axes
import matplotlib.pyplot as plt
from PIL import Image


class VizManager:
    @staticmethod
    def individual_images(
        images: List[Image.Image],
        titles: List[str],
        ncols: int,
        cell_size: Tuple[int, int],
    ) -> None:
        """Plots a list of images in a grid."""
        plot_image_grid(images, titles, ncols, cell_size)

    @staticmethod
    def image_sets(
        image_sets: List[List[Image.Image]],
        title_sets: List[List[str]],
        ncols: int,
        cell_size: Tuple[int, int],
    ) -> None:
        for i, s in enumerate(image_sets):
            print(f"Set: {i}")
            plot_image_grid(s, title_sets[i], ncols, cell_size)


def set_image_on_axes(image: Image.Image, ax: matplotlib.axes.Axes, title: str) -> None:
    cmap = "gray" if image.mode == "L" else None
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(title, fontsize=7)
    ax.imshow(image, cmap=cmap, vmin=0, vmax=255)


def plot_image_grid(
    images: List[Image.Image], titles: List[str], ncols: int, cell_size: Tuple[int, int]
) -> None:
    nrows = math.ceil(len(images) / ncols)
    ncols = min(ncols, len(images))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(cell_size[0] * ncols, cell_size[1] * nrows)
    )
    if nrows > 1:
        idx = 0
        for i in range(nrows):
            for j in range(ncols):
                idx = i * ncols + j
                if idx >= len(images):
                    axes[i, j].axis("off")
                    continue
                set_image_on_axes(images[idx], axes[i, j], titles[idx])
            if idx >= len(images):
                break
    elif ncols > 1:
        for i in range(min(ncols, len(images))):
            set_image_on_axes(images[i], axes[i], titles[i])
    else:
        set_image_on_axes(images[0], axes, titles[0])
    plt.show()
