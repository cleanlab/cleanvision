import math, hashlib, imagehash
from PIL import ImageStat, ImageFilter
import numpy as np


def get_brightness_score(img):
    """
    Scores the overall brightness for a given image to find ones that are too bright and too dark


    Parameters
    ----------
    img: PIL image
    a PIL image object for which the brightness score is calculated


    Returns
    -------
    bright_score: float
    a float between 0 and 1 representing if the image suffers from being too bright or too dark
    a lower number means a more severe issue
    """
    stat = ImageStat.Stat(img)
    try:
        r, g, b = stat.mean
    except:
        r, g, b = (
            stat.mean[0],
            stat.mean[0],
            stat.mean[0],
        )  # deals with black and white images
        # print(f"WARNING: {img} does not have just r, g, b values")
    cur_bright = (
        math.sqrt(0.241 * (r**2) + 0.691 * (g**2) + 0.068 * (b**2))
    ) / 255
    return cur_bright
    # bright_score = min(cur_bright, 1 - cur_bright)  # too bright or too dark
    # return bright_score


def check_odd_size(img):
    """
    Scores the proportions for a given image to find ones with odd sizes


    Parameters
    ----------
    img: PIL image
    a PIL image object for which the size score is calculated


    Returns
    -------
    prop_score: float
    a float between 0 and 1 representing if the image suffers from being having an odd size
    a lower number means a more severe issue
    """
    width, height = img.size
    size_score = min(width / height, height / width)  # consider extreme shapes
    return size_score


def check_entropy(img):
    """
    Scores the entropy for a given image to find ones that are potentially occluded.


    Parameters
    ----------
    img: PIL image
    a PIL image object for which the entropy score is calculated


    Returns
    -------
    entropy_score: float
    a float between 0 and 1 representing the entropy of an image
    a lower number means potentifal occlusion
    """
    entropy = img.entropy()
    return entropy


def check_static(img):
    """
    Calls check_entropy to get images that may be static images


    Parameters
    ----------
    img: PIL image
    a PIL image object for which the entropy score is calculated


    Returns
    -------
    static_score: float
    a float between 0 and 1 representing the 1-entropy of an image
    a lower number means potential static image
    """
    return 1 - check_entropy(img)


def check_blurriness(img):
    """
    Scores the overall blurriness for a given image


    Parameters
    ----------
    img: PIL image
    a PIL image object for which the brightness score is calculated


    Returns
    -------
    blur_score: int
    an integer score where 0 means image is blurry, 1 otherwise
    """
    # todo improve this
    img = img.convert("L")  # Convert image to grayscale

    # Calculating Edges using the Laplacian Kernel
    final = img.filter(
        ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8, -1, -1, -1, -1), 1, 0)
    )
    out = ImageStat.Stat(final).var[0]
    return out


def get_image_hash(img):
    """
    Updates hash information for the set of images to find duplicates

    Parameters
    ----------
    img: PIL image
    a PIL image object for which the hash is calculated

    image_name: str
    a string representing the image name

    count: int
    an integer representing the current image index in the dataset

    issue_info: dict[str, list[int]]
    the ImageDataset attribute self.issue_info where the key is the issue checked by this function ("Duplicated")
    and the value is a list of all the indices with this issue

    misc_info: dict[str, list]
    the ImageDataset attribute self.misc_info

    Returns
    -------
    (issue_info, misc_info): tuple

    a tuple of the dictionaries updated with new information given by img

    """

    img_hash = hashlib.md5(img.tobytes()).hexdigest()
    return img_hash


def get_near_duplicate_hash(img, **kwargs):
    """
    Updates hash information for the set of images to find duplicates


    Parameters
    ----------
    img: PIL image
    a PIL image object for which the hash is calculated

    image_name: str
    a string representing the image name

    count: int
    an integer representing the current image index in the dataset

    issue_info: dict[str, list[int]]
    the ImageDataset attribute self.issue_info where the key is the issue checked by this function ("Duplicated")
    and the value is all the indices with this issue

    misc_info: dict[str, list]
    the ImageDataset attribute self.misc_info

    kwargs: dict
        Keyword arguments specifying:
        what type of image hash to compare between images in order to decide near-duplicates,
        and other configuration-settings of this hash function (eg. the size of the output hash)

        Possible kwargs include:
            `hash_type`: (str) type of hash to use.
            `hash_size`: (int) size of hash to use.
            TODO:list possibilities or point to link with them.

    Returns
    -------
    (issue_info, misc_info): tuple

    a tuple of the dictionaries updated with new information given by img

    """
    img_hash = imagehash.whash(img, hash_size=8)
    return img_hash
    # HASHTYPES = {"whash": imagehash.whash}
    # DEFAULT_HASHTYPE = "whash"
    # DEFAULT_HASHSIZE = 8
    # hash_size = kwargs.get("hash_size", DEFAULT_HASHSIZE)
    # if not (isinstance(hash_size, int) and hash_size > 0):
    #     raise ValueError("Invalid `hash_size` specified in kwargs, must be positive integer.")
    # hash_type = kwargs.get("hash_type", DEFAULT_HASHTYPE)  # [TODO] kwargs not handled correctly
    # if hash_type in HASHTYPES:
    #     hash_function = HASHTYPES[hash_type]
    # else:
    #     raise ValueError(f"Invalid `hash_type` specificed in kwargs, must be one of: {HASHTYPES.keys()}")
    #
    # if "Near Duplicates" not in issue_info:
    #     issue_info["Near Duplicates"] = []
    #     misc_info["Near Duplicate Imagehashes"] = set()
    #     misc_info["Imagehash to Image"] = {}
    #     misc_info["Near Duplicate Image Groups"] = {}
    #
    # cur_hash = hash_function(img, hash_size=hash_size)
    # if cur_hash in misc_info["Near Duplicate Imagehashes"]:
    #     misc_info["Imagehash to Image"][cur_hash].append(count)
    #     imgs_with_cur_hash = misc_info["Imagehash to Image"][cur_hash]
    #     if len(imgs_with_cur_hash) >= 2:  # a near-duplicate group
    #         misc_info["Near Duplicate Image Groups"][cur_hash] = imgs_with_cur_hash
    # else:
    #     misc_info["Near Duplicate Imagehashes"].add(cur_hash)
    #     misc_info["Imagehash to Image"][cur_hash] = [count]
    # issue_info["Near Duplicates"] = list(misc_info["Near Duplicate Image Groups"].values())
    # return (issue_info, misc_info)


def check_grayscale(im):  # return 1 if grayscale else 0
    imarr = np.asarray(im)
    if len(imarr.shape) == 2 or im.mode == "L":
        return 1
    elif len(imarr.shape) == 3 or im.mode == "RGB":
        rgb_channels = imarr.reshape(-1, 3).T
        return 1 if (np.diff(rgb_channels, axis=0) == 0).all() else 0
    else:
        raise ValueError("Cannot check images other than grayscale or RGB")


def find_hot_pixels(im):
    imarr = np.asarray(im.convert("L"))
    blurred = imarr.filter(ImageFilter.MedianFilter(size=2))

    diff = imarr - blurred
    threshold = 10 * np.std(diff)
    num_hot_pixels = (np.abs(diff[1:-1, 1:-1]) > threshold).sum()
    return num_hot_pixels
