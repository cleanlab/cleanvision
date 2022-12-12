import statistics, os, glob
from collections import OrderedDict
import numpy as np

TYPES = [
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.jp2",
    "*.TIFF",
    "*.WebP",
    "*.PNG",
    "*.JPEG",
    "*.png",
]  # filetypes supported by PIL


def get_sorted_images(
    path,
    verbose=True,
):
    """
    Used in initialization of ImageDataset Class
    Obtains image files of supported types and
    sorts them based on filenames numerically and alphabetically


    Parameters
    ----------
    path: str (an attribute of ImageDataset Class)
    a string represening the current working directory


    Returns
    -------
    sorted_names: list[str]
    a list of image filenames sorted numerically and alphabetically
    """
    # if not os.path.isdir(path):  # check if specified path is an existing directory
    #     raise Exception(f"The current path {path} is not valid.")
    image_file_names = []
    for type in TYPES:
        filetype_images = glob.glob(os.path.join(path, type), recursive=True)
        if filetype_images == []:
            continue
        image_file_names += filetype_images
    # base_image_names = []
    # for r in image_file_names:
    #     base_image_names.append(os.path.basename(r))  # extract image name
    return sorted(image_file_names)  # sort image names alphabetically and numerically


def get_zscores(scores):
    mean = np.mean(scores)
    stdev = np.std(scores)
    zscores = (scores - mean) / stdev
    return zscores


def get_is_issue(scores, threshold):
    threshold_score = np.percentile(scores, threshold)
    return scores < threshold_score


def analyze_scores_old(scores, threshold):
    """
    Analyzes the scores for a given issue check,
    including computing the z-scores (where 2 standard deviations left of mean is considered as significant)
    and sorting image indices based on severity of issue

    Parameters
    ----------
    scores: list[float]
    a list of scores for a particular issue check ordered by image order (index in list corresponds to image index)

    threshold: int
    an integer representing the percentile threshold where all scores strictly lower than this threshold are issues
    default set to 5 from find_issues() method of ImageDataset
    Returns
    -------
    (issue_indices, issue_bool): tuple

    issue_indices: list[int]
    a list of images indices suffering from the given issue ordered by severity (high to low)

    issue_bool: dict[int, bool]
    a dictionary where keys are image indices in ascending order, and respective values are binary integers
    1 if the image suffers from the given issue
    0 otherwise

    sorted_zscores: dict[int, float]
    a dictionary sorted based on values from low to high
    where keys are image indices, and respective values are z-scores
    """
    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores)
    threshold_score = np.percentile(scores, threshold)
    scores_dict = {}  # stores index and scores
    for i, val in enumerate(scores):
        scores_dict[i] = val
    sorted_scores = {
        k: v for k, v in sorted(scores_dict.items(), key=lambda item: item[1])
    }  # sort scores from low to high (worse to better images)
    sorted_zscores = {k: (v - mean) / stdev for k, v in sorted_scores.items()}
    issue_indices = []  # high to low severity
    issue_bool = (
        OrderedDict()
    )  # ascending image indices, boolean to denote if issue present
    for k1, v1 in sorted_scores.items():
        if v1 < threshold_score:
            issue_indices.append(k1)
    for (
        k2,
        v2,
    ) in (
        scores_dict.items()
    ):  # ascending indices order for images, label if an image suffers from an issue
        if v2 < threshold_score:
            issue_bool[k2] = 1
        else:
            issue_bool[k2] = 0
    return (issue_indices, issue_bool, sorted_zscores)


def display_images(indices, num_preview):
    """
    Used in initialization of ImageDataset Class
    Sorts image files based on image filenames numerically and alphabetically


    Parameters
    ----------
    indices: list
    a flat list or one level nested list containing the indices of images with a given issue

    num_preview: int
    if indices is a flat list: an integer representing the number of images with the issue shown
    if indices is a nested list: an integer representing the number of issue image groups shown

    Returns
    -------
    A flat list with length num_preview, containing indices of images displayed to user
    """
    outlen = min(num_preview, len(indices))
    print("outlen", outlen)
    if type(indices[0]) == list:
        out = []
        for i in range(outlen):
            out += indices[i]
        return out
    else:
        return indices[:outlen]
