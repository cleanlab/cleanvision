import glob
import multiprocessing
import os
from typing import Dict, List, Any

import pandas as pd

from cleanvision.utils.constants import IMAGE_FILE_EXTENSIONS

# psutil is a package used to count physical cores for multiprocessing
# This package is not necessary, because we can always fall back to logical cores as the default
try:
    import psutil

    PSUTIL_EXISTS = True
except ImportError:  # pragma: no cover
    PSUTIL_EXISTS = False


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

    abs_dir_path = os.path.abspath(os.path.expanduser(dir_path))
    print(f"Reading images from {abs_dir_path}")
    filepaths = []
    for ext in IMAGE_FILE_EXTENSIONS:
        filetype_images = glob.glob(
            os.path.join(abs_dir_path, "**", ext), recursive=True
        )
        if len(filetype_images) == 0:
            continue
        filepaths += filetype_images

    # glob.glob is case-insensitive in Windows and picks up the same file twice while looking for *.png and *.PNG.
    unique_filepaths = list(set(filepaths))
    return sorted(unique_filepaths)  # sort image names alphabetically and numerically


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


def get_is_issue_colname(issue_type: str) -> str:
    return f"is_{issue_type}_issue"


def get_score_colname(issue_type: str) -> str:
    return f"{issue_type}_score"


def update_df(
    df: pd.DataFrame, new_df: pd.DataFrame, overwrite: bool = True
) -> pd.DataFrame:
    """Updates `df` with columns of `new_df` (must have same number of rows).
    New columns in `new_df` not present in `df` are appended as extra columns.
    Columns in `df` that are present in `new_df` have their values overwritten if ``overwrite==True``.
    """
    columns_to_update, new_columns = [], []
    for column in new_df.columns:
        if column in df.columns:
            columns_to_update.append(column)
        else:
            new_columns.append(column)
    if overwrite:
        for column_name in columns_to_update:
            df[column_name] = new_df[column_name]
    df = df.join(new_df[new_columns], how="left")
    return df
