import math
from typing import TypeVar, List, Tuple, Set

import matplotlib.pyplot as plt  # type: ignore
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid  # type: ignore

TVizManager = TypeVar("TVizManager", bound="VizManager")  # self type for the class


class VizManager:
    def __init__(self: TVizManager):
        pass

    @staticmethod
    def individual_images(
        filepaths: List[str], ncols: int, cell_size: Tuple[int, int], cmap=None
    ) -> None:
        plot_image_grid(filepaths, ncols, cell_size, cmap)

    @staticmethod
    def image_sets(
        filepath_sets: Set[List[str]], ncols: int, cell_size: Tuple[int, int]
    ) -> None:
        for s in filepath_sets:
            plot_image_grid(s, ncols, cell_size)


def plot_image_grid(
    filepaths: List[str], ncols: int, cell_size: Tuple[int, int], cmap=None
) -> None:
    nrows = math.ceil(len(filepaths) / ncols)
    ncols = min(ncols, len(filepaths))
    fig = plt.figure(figsize=(cell_size[0] * ncols, cell_size[1] * nrows))

    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.2, share_all=True)

    for ax, path in zip(grid, filepaths):
        # Iterating over the grid returns the Axes.
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_title(path.split("/")[-1], fontsize=5)
        ax.imshow(Image.open(path), cmap=cmap)
    plt.show()
