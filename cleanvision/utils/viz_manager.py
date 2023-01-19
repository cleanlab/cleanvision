import math

import matplotlib.pyplot as plt  # type: ignore
from PIL import Image  # type: ignore
from mpl_toolkits.axes_grid1 import ImageGrid  # type: ignore


class VizManager:
    def __init__(self):
        pass

    @staticmethod
    def individual_images(filepaths, ncols, cell_size):
        plot_image_grid(filepaths, ncols, cell_size)

    @staticmethod
    def image_sets(filepath_sets, ncols, cell_size):
        for s in filepath_sets:
            plot_image_grid(s, ncols, cell_size)


def plot_image_grid(filepaths, ncols, cell_size):
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
        ax.imshow(Image.open(path))
    plt.show()
