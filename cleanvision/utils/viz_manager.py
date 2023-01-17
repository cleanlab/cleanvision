import math

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from mpl_toolkits.axes_grid1 import ImageGrid


class VizManager:
    def __init__(self):
        pass

    @staticmethod
    def individual_images(filepaths, ncols, fig_width):
        plot_image_grid(filepaths, ncols, fig_width)

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


# todo find a way to show PIL images sequentially
def concat_images(image_paths, size, shape=None):
    # Open images and resize them
    width, height = size
    images = map(Image.open, image_paths)
    images = [ImageOps.fit(image, size, Image.ANTIALIAS) for image in images]

    # Create canvas for the final image with total size
    shape = shape if shape else (1, len(images))
    image_size = (width * shape[1], height * shape[0])
    image = Image.new("RGB", image_size)

    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col, height * row
            idx = row * shape[1] + col
            image.paste(images[idx], offset)

    return image
