import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid


class VizManager:
    def __init__(self):
        pass

    @staticmethod
    def property_based(filepaths, nrows, ncols, figsize):
        fig = plt.figure(figsize=figsize)

        grid = ImageGrid(
            fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.2, share_all=True
        )

        for ax, path in zip(grid, filepaths):
            # Iterating over the grid returns the Axes.
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.set_title(path.split("/")[-1], fontsize=5)
            ax.imshow(Image.open(path))

        plt.show()


VIZ_REGISTRY = {"property_based": VizManager.property_based}
