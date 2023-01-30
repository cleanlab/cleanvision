import glob
import os

from typing import Dict, List, Any

import numpy as np

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


def get_filepaths(
    dir_path: str,
) -> List[str]:
    """
    Used in initialization of ImageDataset Class
    Obtains image files of supported types and
    sorts them based on filenames numerically and alphabetically


    Parameters
    ----------
    dir_path: str (an attribute of ImageDataset Class)
    a string represening the current working directory


    Returns
    -------
    sorted_names: list[str]
    a list of image filenames sorted numerically and alphabetically
    """
    # if not os.path.isdir(path):  # check if specified path is an existing directory
    #     raise Exception(f"The current path {path} is not valid.")
    filepaths = []
    for type in TYPES:
        filetype_images = glob.glob(os.path.join(dir_path, type), recursive=True)
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
