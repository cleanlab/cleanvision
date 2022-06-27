import math, hashlib, imagehash, copy
from PIL import ImageStat


def check_brightness(img):
    """
    Calculates the brightness score for a given image


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


def check_odd_size(img):
    """
    Calculates the proportions score for a given image to find odd image sizes


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
    Calculates the entropy score for a given image to find potentially occluded images


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


def check_duplicated(img, image_name, count, issue_info, misc_info):
    """
    Updates hash information for the set of images to find duplicates


    Parameters
    ----------
    img: PIL image
    a PIL image object for which the hash is calculated

    image_name: string
    a string representing the image name

    count: int
    an integer representing the current image index in the dataset

    issue_info: dict
    the ImageDataset attribute self.issue_info where the key is the issue checked by this function ("Duplicated")
    and the value is all the indices with this issue

    misc_info: dict
    the ImageDataset attribute self.misc_info

    Returns
    -------
    a tuple: (issue_info, misc_info)

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
        if len(imgs_with_cur_hash) == 2:  # found a duplicate pair
            misc_info["Duplicate Image Groups"][cur_hash] = imgs_with_cur_hash
        elif (
            len(misc_info["Hash to Image"][cur_hash]) > 2
        ):  # more than 2 of the same images
            misc_info["Duplicate Image Groups"][cur_hash].append(image_name)
    else:
        misc_info["Image Hashes"].add(cur_hash)
        misc_info["Hash to Image"][cur_hash] = [image_name]
    return (issue_info, misc_info)


def check_near_duplicates(img, image_name, count, issue_info, misc_info):
    """
    Updates imagehash (wavelet hashing) information for the set of images to find near-duplicates

    Parameters
    ----------
    img: PIL image
    a PIL image object for which the brightness score is calculated

    image_name: string
    a string representing the image name

    count: int
    an integer representing the current image index in the dataset

    issue_info: dict
    the ImageDataset attribute self.issue_info where the key is the issue checked by this function ("Near duplicates")
    and the value a nested list where each sublist is a group of near duplicates

    misc_info: dict
    the ImageDataset attribute self.misc_info

    Returns
    -------
    a tuple: (issue_info, misc_info)

    a tuple of the dictionaries updated with new information given by img

    """
    cur_imagehash = imagehash.whash(img)
    if "Near duplicates" not in issue_info:
        misc_info["Near duplicates tuples"] = []
        issue_info["Near duplicates"] = []
    modify = False  # whether img is a near duplicate that requires modification of issue_info
    alone = True  # image is not a near-duplicate of any previously checked images
    for index in range(len(misc_info["Near duplicates tuples"])):
        group = misc_info["Near duplicates tuples"][index]
        near_count = 0
        group_len = len(group)
        temp_group_tuple = copy.deepcopy(group)
        for img_and_hash in group:
            if img_and_hash[0] - cur_imagehash < 8:
                alone = False  # found a near duplicates pair
                near_count += 1
        if near_count == group_len:  # near duplicates with every image in a group
            temp_group_tuple.append((cur_imagehash, count))
            misc_info["Near duplicates tuples"][index] = temp_group_tuple
            modify = True
    if alone:  # img is not a near duplicate of any seen images
        misc_info["Near duplicates tuples"].append([(imagehash.whash(img), count)])
    if modify:
        issue_info["Near duplicates"] = [
            [tup[1] for tup in L]
            for L in misc_info["Near duplicates tuples"]
            if len(L) > 1
        ]
    return (issue_info, misc_info)
