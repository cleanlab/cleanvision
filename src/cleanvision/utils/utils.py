import glob
import os

from typing import Dict, List, Any

TYPES: List[str] = [
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


# todo: make recursive an option
def get_filepaths(
    dir_path: str,
) -> List[str]:
    """Gets paths of all image files in the dir_path recursively.
     All image files with extension in TYPES are allowed.
     Returns a sorted list of sorted filepaths


    Parameters
    ----------
    dir_path: str
        Path to the dir containing image files, can be relative or absolute path


    Returns
    -------
    List[str]
        Sorted list of image filepaths, note that all paths in this list are absolute paths
    """

    if not os.path.isdir(dir_path):
        raise NotADirectoryError

    abs_dir_path = os.path.abspath(dir_path)
    print(f"Reading images from {abs_dir_path}")
    filepaths = []
    for type in TYPES:
        filetype_images = glob.glob(os.path.join(abs_dir_path, type), recursive=True)
        if len(filetype_images) == 0:
            continue
        filepaths += filetype_images
    return sorted(filepaths)  # sort image names alphabetically and numerically


def deep_update_dict(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """Updates nested dictionary

    Parameters
    ----------
    d : dict
        dictionary to update
    u : dict
        Updates

    Returns
    -------
    dict
        Updated dictionary
    """
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_is_issue_colname(issue_type):
    return f"is_{issue_type}_issue"
