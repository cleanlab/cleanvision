import math, hashlib, imagehash, copy
from PIL import ImageStat, ImageFilter
import numpy as np



def check_brightness(img, **kwargs):
    """
    Scores the overall brightness for a given image to find ones that are too bright and too dark
    generates 'Brightness sorted z-scores' in images.misc_info 

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
        print(f"WARNING: {img} does not have just r, g, b values")
    cur_bright = (
        math.sqrt(0.241 * (r**2) + 0.691 * (g**2) + 0.068 * (b**2))
    ) / 255
    bright_score = min(cur_bright, 1 - cur_bright)  # too bright or too dark
    return bright_score


def check_odd_size(img, **kwargs):
    """
    Scores the proportions for a given image to find ones with odd sizes
    generates 'Odd Size sorted z-scores' in images.misc_info

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


def check_entropy(img, **kwargs):
    """
    Scores the entropy for a given image to find ones that are potentially occluded. 
    generates 'Potential Occlusion sorted z-scores' in images.misc_info

    Parameters
    ----------
    img: PIL image
    a PIL image object for which the entropy score is calculated


    Returns
    -------
    entropy_score: float
    a float between 0 and 1 representing the entropy of an image
    a lower number means potential occlusion
    """
    entropy_score = img.entropy() / 10
    return entropy_score

def check_static(img, **kwargs):
    """
    Calls check_entropy to get images that may be static images
    generates 'Potential Static sorted z-scores' in images.misc_info

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
    return 1-check_entropy(img)

def check_blurriness(img, **kwargs):
    """
    Scores the overall blurriness for a given image
    generates 'Blurry sorted z-scores' in images.misc_info

    Parameters
    ----------
    img: PIL image
    a PIL image object for which the brightness score is calculated


    Returns
    -------
    blur_score: int
    an integer score where 0 means image is blurry, 1 otherwise
    """
    img = img.convert("L") #Convert image to grayscale
    # Calculating Edges using the Laplacian Kernel
    final = img.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
                                            -1, -1, -1, -1), 1, 0))
    out = ImageStat.Stat(final).var[0]
    blur_score = 1/(1+(math.e)**(2*(-out+260))) #calculate score between 0 & 1 using modified sigmoid function
    return blur_score


def check_duplicated(img, image_name, count, issue_info, misc_info, **kwargs):
    """
    Updates hash information for the set of images to find duplicates
    generates 
    'Image Hashes',
    'Hash to Image',
    'Duplicate Image Groups'
    in images.misc_info

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
    if "Duplicated" not in issue_info:
        issue_info["Duplicated"] = []
        misc_info["Image Hashes"] = set()
        misc_info["Hash to Image"] = {}
        misc_info["Duplicate Image Groups"] = {}
    cur_hash = hashlib.md5(img.tobytes()).hexdigest()
    if cur_hash in misc_info["Image Hashes"]:
        issue_info["Duplicated"].append(count)
        misc_info["Hash to Image"][cur_hash].append(image_name)
        imgs_with_cur_hash = misc_info["Hash to Image"][cur_hash]
        if len(imgs_with_cur_hash) >= 2:  # found a duplicate pair
            misc_info["Duplicate Image Groups"][cur_hash] = imgs_with_cur_hash
    else:
        misc_info["Image Hashes"].add(cur_hash)
        misc_info["Hash to Image"][cur_hash] = [image_name]
    return (issue_info, misc_info)

def check_near_duplicates(img, image_name, count, issue_info, misc_info, **kwargs):
    """
    Updates hash information for the set of images to find duplicates
    generates 
    'Near Duplicate Imagehashes',
    'Imagehash to Image',
    'Near Duplicate Image Groups'
    in images.misc_info

    kwargs: 
    "hashtype"= "whash", "phash", "color_hash", "ahash"
    "hash_size" = int
    
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

    Returns
    -------
    (issue_info, misc_info): tuple

    a tuple of the dictionaries updated with new information given by img

    """
    hashtypes = {"whash": imagehash.whash, "phash": imagehash.phash, "colorhash": imagehash.colorhash, "ahash": imagehash.average_hash}
    if "Near Duplicates" not in issue_info:
        issue_info["Near Duplicates"] = []
        misc_info["Near Duplicate Imagehashes"] = set()
        misc_info["Imagehash to Image"] = {}
        misc_info["Near Duplicate Image Groups"] = {}
    if kwargs: #hash function specified by user
        hash_size = kwargs["hash_size"]
        cur_hash = hashtypes[kwargs["hashtype"]](img, hash_size)
    else: 
        cur_hash = imagehash.phash(img, hash_size = 8)
    if cur_hash in misc_info["Near Duplicate Imagehashes"]:
        misc_info["Imagehash to Image"][cur_hash].append(count)
        imgs_with_cur_hash = misc_info["Imagehash to Image"][cur_hash]
        if len(imgs_with_cur_hash) >= 2:  # a near-duplicate group
            misc_info["Near Duplicate Image Groups"][cur_hash] = imgs_with_cur_hash
    else:
        misc_info["Near Duplicate Imagehashes"].add(cur_hash)
        misc_info["Imagehash to Image"][cur_hash] = [count]
    issue_info["Near Duplicates"] = list(misc_info["Near Duplicate Image Groups"].values())
    return (issue_info, misc_info)
