from typing import List, Tuple, Dict

import math
import matplotlib.axes
import matplotlib.pyplot as plt
from PIL import Image


class VizManager:
    @staticmethod
    def individual_images(
        images: List[Image.Image],
        title_info: Dict[str, List[str]],
        ncols: int,
        cell_size: Tuple[int, int],
    ) -> None:
        """Plots a list of images in a grid."""
        plot_image_grid(images, title_info, ncols, cell_size)

    @staticmethod
    def image_sets(
        image_sets: List[List[Image.Image]],
        title_info_sets: List[Dict[str, List[str]]],
        ncols: int,
        cell_size: Tuple[int, int],
    ) -> None:
        for i, s in enumerate(image_sets):
            print(f"Set: {i}")
            plot_image_grid(s, title_info_sets[i], ncols, cell_size)


def set_image_on_axes(image: Image.Image, ax: matplotlib.axes.Axes, title: str) -> None:
    cmap = "gray" if image.mode == "L" else None
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(title, fontsize=7)
    ax.imshow(image, cmap=cmap, vmin=0, vmax=255)


def truncate_titles(cell_width: int, titles: List[str]) -> List[str]:
    """Converts font size of 7 into inches"""
    CHARACTER_SIZE_INCHES = 7 * (1 / 72)

    chars_allowed = math.ceil(cell_width / CHARACTER_SIZE_INCHES) - 4

    k1 = 1
    while k1 <= chars_allowed and titles[0][:k1] == titles[1][:k1]:
        k1 += 1
    k2 = 1
    while (
        k2 <= chars_allowed
        and titles[0][(len(titles[0]) - k2) :] == titles[1][(len(titles[1]) - k2) :]
    ):
        k2 += 1

    if k1 > k2:
        truncate_from_front = True
    else:
        truncate_from_front = False

    for i in range(len(titles)):
        title_width = len(titles[i]) * CHARACTER_SIZE_INCHES
        if title_width >= cell_width:
            titles[i] = (
                ("..." + titles[i][len(titles[i]) - chars_allowed :])
                if truncate_from_front
                else (titles[i][:chars_allowed] + "...")
            )
    return titles


def construct_titles(title_info: Dict[str, List[str]], cell_width: int) -> List[str]:
    keys = list(title_info.keys())
    nimages = len(title_info[keys[0]])

    # truncate longer lines
    if nimages > 1:
        for key in keys:
            title_info[key] = truncate_titles(cell_width, title_info[key])

    # join all keys
    titles = []
    for i in range(nimages):
        titles.append("\n".join(title_info[key][i] for key in keys))
    return titles


def plot_image_grid(
    images: List[Image.Image],
    title_info: Dict[str, List[str]],
    ncols: int,
    cell_size: Tuple[int, int],
) -> None:
    nrows = math.ceil(len(images) / ncols)
    ncols = min(ncols, len(images))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(cell_size[0] * ncols, cell_size[1] * nrows)
    )
    titles = construct_titles(title_info, cell_size[0])
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
