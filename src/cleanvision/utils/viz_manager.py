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
    
    # Convert list of images to a 4D Numpy array
    arr = np.array([np.array(image) for image in images])
    
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(cell_size[0] * ncols, cell_size[1] * nrows)
    )

    # Create a 2D array of indices
    idxs = np.arange(nrows * ncols).reshape(nrows, ncols)

    # Set axes properties
    for ax in axes.flatten():
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # Set images on axes using advanced indexing
    axes[idxs[:len(images) // ncols + 1, :len(images) % ncols]] = arr
    for i, title in enumerate(titles):
        axes.flat[i].set_title(title, fontsize=7)

    plt.show()
