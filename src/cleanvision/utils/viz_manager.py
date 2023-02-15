import math
from typing import List, Optional, Tuple, Set, Union

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1 import ImageGrid  # type: ignore


class VizManager:
    @staticmethod
    def individual_images(
        filepaths: List[str],
        ncols: int,
        cell_size: Tuple[int, int],
        cmap: Optional[Union[str, Colormap]] = None,
    ) -> None:
        plot_image_grid(filepaths, ncols, cell_size, cmap)

    @staticmethod
    def image_sets(
        filepath_sets: Set[List[str]], ncols: int, cell_size: Tuple[int, int]
    ) -> None:
        for s in filepath_sets:
            plot_image_grid(s, ncols, cell_size)


def set_image_on_axes(path, ax):
    image = Image.open(path)
    cmap = "gray" if image.mode == "L" else None
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(path.split("/")[-1], fontsize=7)
    ax.imshow(image, cmap=cmap)


def plot_image_grid(
    filepaths: List[str],
    ncols: int,
    cell_size: Tuple[int, int],
    cmap: Optional[Union[str, Colormap]] = None,
) -> None:
    nrows = math.ceil(len(filepaths) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(cell_size[0] * ncols, cell_size[1] * nrows)
    )
    if nrows > 1:
        idx = 0
        for i in range(nrows):
            for j in range(ncols):
                idx = i * ncols + j
                if idx >= len(filepaths):
                    break
                set_image_on_axes(filepaths[idx], axes[i, j])
            if idx >= len(filepaths):
                break
    else:
        for i in range(ncols):
            if i < len(filepaths):
                set_image_on_axes(filepaths[i], axes[i])
    plt.show()
