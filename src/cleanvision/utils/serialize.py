from __future__ import annotations

import os
import pickle
import warnings
from typing import TYPE_CHECKING

import pandas as pd

import cleanvision
from cleanvision.dataset.hf_dataset import HFDataset

if TYPE_CHECKING:  # pragma: no cover
    from cleanvision.imagelab import Imagelab

# Constants:
OBJECT_FILENAME = "imagelab.pkl"
ISSUES_FILENAME = "issues.csv"
ISSUE_SUMMARY_FILENAME = "summary.csv"


class _Serializer:
    @staticmethod
    def _save_issues(path: str, imagelab: Imagelab) -> None:
        """Saves the issues to disk."""
        issues_path = os.path.join(path, ISSUES_FILENAME)
        imagelab.issues.to_csv(issues_path, index=False)

        issue_summary_path = os.path.join(path, ISSUE_SUMMARY_FILENAME)
        imagelab.issue_summary.to_csv(issue_summary_path, index=False)

    @staticmethod
    def _validate_version(imagelab: Imagelab) -> None:
        current_version = cleanvision.__version__  # type: ignore[attr-defined]
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

        # Save the imagelab object to disk.
        with open(os.path.join(path, OBJECT_FILENAME), "wb") as f:
            pickle.dump(imagelab, f)

        print(f"Saved Imagelab to folder: {path}")
        print(
            "The data path and dataset must be not be changed to maintain consistent state when loading this Imagelab"
        )

    @classmethod
    def deserialize(cls, path: str) -> Imagelab:
        """Deserializes the imagelab object from disk."""

        if not os.path.exists(path):
            raise ValueError(f"No folder found at specified path: {path}")

        with open(os.path.join(path, OBJECT_FILENAME), "rb") as f:
            imagelab: Imagelab = pickle.load(f)

        cls._validate_version(imagelab)

        # Load the issues from disk.
        issues_path = os.path.join(path, ISSUES_FILENAME)
        if os.path.exists(issues_path):
            imagelab.issues = pd.read_csv(issues_path)
        else:
            raise ValueError(f"Could not find {ISSUES_FILENAME} at specified path")

        issue_summary_path = os.path.join(path, ISSUE_SUMMARY_FILENAME)
        if os.path.exists(issue_summary_path):
            imagelab.issue_summary = pd.read_csv(issue_summary_path)
        else:
            raise ValueError(
                f"Could not find {ISSUE_SUMMARY_FILENAME} at specified path"
            )

        print("Successfully loaded Imagelab")
        return imagelab
