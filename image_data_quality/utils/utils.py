import statistics, os, glob
from collections import OrderedDict
from PIL import Image
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
]  # filetypes supported by PIL


def get_sorted_images(
    path,
):  
    """
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
    if not os.path.isdir(path):  # check if specified path is an existing directory
        raise Exception(f"The current path {path} is not valid.")
    image_file_names = []
    for type in TYPES:
        filetype_images = glob.glob(os.path.join(path, type))
        if filetype_images == []:
            continue
        image_file_names += filetype_images
    base_image_names = []
    for r in image_file_names:
        base_image_names.append(os.path.basename(r))  # extract image name
    return sorted(base_image_names)  # sort image names alphabetically and numerically


def analyze_scores(scores, threshold):
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
    num_images = len(scores)
    scores_array = np.array(scores)
    mean = np.mean(scores_array)
    stdev = np.std(scores_array)
    threshold_score = np.percentile(scores_array, threshold)
    scores_nested = [] # stores index and scores
    zscores_nested = []
    for i, val in np.ndenumerate(scores_array):
        scores_nested.append([i[0], val])
        zscores_nested.append([i[0],(val - mean) / stdev])
    scores_array = np.array(scores_nested)
    zscores_array = np.array(zscores_nested)
    scores_array_sorted = scores_array[scores_array[:, 1].argsort()] # sort scores from low to high (worse to better images)
    zscores_array_sorted = zscores_array[zscores_array[:, 1].argsort()]
    issue_indices = []  # high to low severity
    issue_bool = (
        OrderedDict()
    )  # ascending image indices, boolean to denote if issue present
    scores_filtered = scores_array_sorted[scores_array_sorted[:,1] < threshold_score]
    for i in scores_filtered:
        issue_indices.append(int(i[0]))
    for n in range(num_images):# ascending indices order for images, label if an image suffers from an issue
        if n in issue_indices:
            issue_bool[n] = 1
        else:
            issue_bool[n] = 0
    return (issue_indices, issue_bool, list(zscores_array_sorted))

def display_images(indices, num_preview):
    '''
    Takes in a flat list or a nested list and 
    returns a flat list of image indices to be displayed


    Parameters
    ----------
    indices: list
    a flat list or one level nested list containing the indices of images with a given issue

    num_preview: int
    if indices is a flat list: an integer representing the number of images with the issue shown
    if indices is a nested list: an integer representing the number of issue image groups shown
    if num_preview is greater than len(indices), show all images with issue. 

    Returns
    -------
    A flat list with length num_preview, containing indices of images displayed to user
    '''
    outlen = min(num_preview, len(indices))
    if type(indices[0])==int: #if flat list
        return indices[:outlen]
    else: #if nested list
        return [item for i in range(outlen) for item in indices[i]]
