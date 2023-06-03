from __future__ import annotations

import os
import pickle
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

import pandas as pd

import cleanvision

if TYPE_CHECKING:  # pragma: no cover
    from cleanvision import Imagelab

# Constants:
OBJECT_FILENAME = "imagelab.pkl"
ISSUES_FILENAME = "issues.csv"
ISSUE_SUMMARY_FILENAME = "issue_summary.csv"


class Serializer:
    @staticmethod
    def _save_issues(path: str, imagelab: Imagelab) -> None:
        """Saves the issues to disk."""
        issues_path = os.path.join(path, ISSUES_FILENAME)
        imagelab.issues.to_csv(issues_path)

        issue_summary_path = os.path.join(path, ISSUE_SUMMARY_FILENAME)
        imagelab.issue_summary.to_csv(issue_summary_path)

    @staticmethod
    def _validate_version(imagelab: Imagelab) -> None:
        current_version = cleanvision.__version__
        imagelab_version = imagelab.cleanvision_version
        if current_version != imagelab_version:
            warnings.warn(
                f"Saved Imagelab was created using different version of cleanvision "
                f"({imagelab_version}) than current version ({current_version}). "
                f"Things may be broken!"
            )

    @classmethod
    def serialize(cls, path: str, imagelab: Imagelab, force: bool) -> None:
        """Serializes the imagelab object to disk.

        Parameters
        ----------
        path : str
            Path to save the imagelab object to.

        imagelab : Imagelab
            The imagelab object to save.

        force : bool
            If True, will overwrite existing files at the specified path.

        Raises
        ------
        FileExistsError
            If `force` is set to False, and an existing path is specified for saving Imagelab instance.

        """
        path_exists = os.path.exists(path)
        if not path_exists:
            os.mkdir(path)
        else:
            if not force:
                raise FileExistsError("Please specify a new path or set force=True")
            print(
                f"WARNING: Existing files will be overwritten by newly saved files at: {path}"
            )

        # Save the issues to disk.
        cls._save_issues(path=path, imagelab=imagelab)

        # clear issues and issue_summary
        imagelab_copy = deepcopy(imagelab)
        imagelab_copy.issues = None
        imagelab_copy.issue_summary = None

        # Save the imagelab object to disk.
        with open(os.path.join(path, OBJECT_FILENAME), "wb") as f:
            pickle.dump(imagelab_copy, f)

        print(f"Saved Imagelab to folder: {path}")
        print(
            "The data path and dataset must be not be changed to maintain consistent state when loading this Imagelab"
        )

    @classmethod
    def deserialize(cls, path: str) -> Imagelab:
        """Deserializes the imagelab object from disk.

        Parameters
        ----------
        path: str
            Path to the saved Imagelab folder previously specified in :py:meth:`_Serializer.serialize` (not the individual pickle file).

        Returns
        -------
        Imagelab

        Raises
        ------
        ValueError:
            If the path specified for imagelab folder does not exist

        """

        if not os.path.exists(path):
            raise ValueError(f"No folder found at specified path: {path}")

        with open(os.path.join(path, OBJECT_FILENAME), "rb") as f:
            imagelab: Imagelab = pickle.load(f)

        cls._validate_version(imagelab)

        # Load the issues from disk.
        issues_path = os.path.join(path, ISSUES_FILENAME)
        imagelab.issues = pd.read_csv(issues_path, index_col=0)

        issue_summary_path = os.path.join(path, ISSUE_SUMMARY_FILENAME)
        imagelab.issue_summary = pd.read_csv(issue_summary_path, index_col=0)

        print("Successfully loaded Imagelab")
        return imagelab
