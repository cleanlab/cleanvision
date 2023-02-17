import glob
import multiprocessing
import os
from typing import Dict, List, Any

# psutil is a package used to count physical cores for multiprocessing
# This package is not necessary, because we can always fall back to logical cores as the default
try:
    import psutil

    PSUTIL_EXISTS = True
except ImportError:  # pragma: no cover
    PSUTIL_EXISTS = False

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


def get_max_n_jobs() -> int:
    n_jobs = None
    if PSUTIL_EXISTS:
        n_jobs = psutil.cpu_count(logical=False)  # physical cores
    if not n_jobs:
        # either psutil does not exist
        # or psutil can return None when physical cores cannot be determined
        # switch to logical cores
        n_jobs = multiprocessing.cpu_count()
    return n_jobs


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

    abs_dir_path = os.path.abspath(dir_path)
    print(f"Reading images from {abs_dir_path}")
    filepaths = []
    for type in TYPES:
        filetype_images = glob.glob(
            os.path.join(abs_dir_path, "**", type), recursive=True
        )
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
