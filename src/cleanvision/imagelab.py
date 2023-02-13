"""
Imagelab is the core class in CleanVision for finding all types of issues in an image dataset.
The methods in this module should suffice for most use-cases,
but advanced users can get extra flexibility via the code in other CleanVision modules.
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Tuple, TypeVar, Type

import numpy as np
import pandas as pd

from cleanvision.issue_managers import (
    IssueType,
    IssueManagerFactory,
    ISSUE_MANAGER_REGISTRY,
)
from cleanvision.utils.base_issue_manager import IssueManager
from cleanvision.utils.constants import (
    IMAGE_PROPERTY,
    DUPLICATE,
    IMAGE_PROPERTY_ISSUE_TYPES_LIST,
    DUPLICATE_ISSUE_TYPES_LIST,
    SETS,
)
from cleanvision.utils.utils import get_filepaths, deep_update_dict
from cleanvision.utils.viz_manager import VizManager

__all__ = ["Imagelab"]

OBJECT_FILENAME = "imagelab.pkl"
TImagelab = TypeVar("TImagelab", bound="Imagelab")


class Imagelab:
    """A single class to find all types of issues in image datasets.

    Parameters
    ----------
    data_path : str
        Path to image files. Imagelab will recursively retrieve all image files from the specified path

    Examples
    --------

    Basic usage of Imagelab class

    .. code-block:: python
        from cleanvision.imagelab import Imagelab
        imagelab = Imagelab("FOLDER_WITH_IMAGES/")
        imagelab.find_issues()
        imagelab.report()


    Attributes
    ----------
    issues : pd.DataFrame
        Dataframe where each row corresponds to an image and columns specify which issues were detected in this image.
        It has two types of columns for each issue type:
        1. <issue_type>_score - This column contains a quality-score for each image for a particular type of issue. 
        Scores are between 0 and 1, lower values indicate images exhibiting more severe instances of this issue.
        2. <issue_type>_bool - This column indicates whether or not the issue_type is detected in each image (a binary decision rather than numeric score).


    issue_summary:
        Dataframe containing summary of all issue types found.
        Specifically, it shows num_images of each issue found in the dataset

    info : dict
        This is a nested dictionary that contains statistics on images or other useful information,
        collected while checking for issues in the dataset.

    Raises
    ------
    ValueError
        If not images are found in the data_path
    """

    def __init__(self, data_path: str) -> None:
        self._filepaths: List[str] = get_filepaths(data_path)
        self._num_images: int = len(self._filepaths)
        if self._num_images == 0:
            raise ValueError(f"No images found in the specified path:{data_path}")
        self.info: Dict[str, Any] = {"statistics": {}}
        self.issue_summary: pd.DataFrame = pd.DataFrame(columns=["issue_type"])
        self.issues: pd.DataFrame = pd.DataFrame(index=self._filepaths)
        self._issue_types: List[str] = []
        self._issue_managers: Dict[str, IssueManager] = {}
        # can be loaded from a file later
        self._config: Dict[str, Any] = self._set_default_config()
        self._path = ""

    def _set_default_config(self) -> Dict[str, Any]:
        """Sets default values for various config variables used in Imagelab class
        The naming convention for methods is {method_name}_{config_variable_name}

        Returns
        -------
        Dict[str, Any]
            Returns a dict with keys as config variables and their values as dict values
        """
        return {
            "visualize_num_images_per_row": 4,
            "report_num_top_issues_values": [3, 5, 10, len(self._issue_types)],
            "report_examples_per_issue_values": [4, 8, 16, 32],
            "report_max_prevalence": 0.5,
            "default_issue_types": [
                IssueType.DARK,
                IssueType.LIGHT,
                IssueType.ODD_ASPECT_RATIO,
                IssueType.LOW_INFORMATION,
                IssueType.EXACT_DUPLICATES,
                IssueType.NEAR_DUPLICATES,
                IssueType.BLURRY,
                IssueType.GRAYSCALE,
            ],
        }

    def list_default_issue_types(self) -> None:
        """Prints a list of all issue types checked by default if no issue types are specified in imagelab.find_issues()"""

        print("Default issue type checked by Imagelab:\n")
        print(
            *[issue_type.value for issue_type in self._config["default_issue_types"]],
            sep="\n",
        )

    def list_possible_issue_types(self) -> None:
        """Prints a list of all possible issue types that can be checked in the dataset.
        It will also include custom added issue types.
        """
        print("All possible issues checked by Imagelab:\n")
        issue_types = {issue_type.value for issue_type in IssueType}
        issue_types.update(ISSUE_MANAGER_REGISTRY.keys())
        print(*issue_types, sep="\n")
        print("\n")

    def _get_issues_to_compute(
        self, issue_types_with_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not issue_types_with_params:
            to_compute_issues_with_params: Dict[str, Any] = {
                issue_type.value: {}
                for issue_type in self._config["default_issue_types"]
            }
        else:
            to_compute_issues_with_params = {
                issue_type_str: params
                for issue_type_str, params in issue_types_with_params.items()
            }
        return to_compute_issues_with_params

    def find_issues(
        self, issue_types: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """Finds issues in the dataset.
        If issue_types is empty or not given, dataset is checked for all default issue types.

        Parameters
        ----------
        issue_types : Dict[str, Any], optional
            Dict with issue_types to check as keys
            The value of this dict is a dict containing hyperparameters for each issue type

        Examples
        --------
        To check for all default issue types use

        .. code-block:: python
            imagelab.find_issues()

        To check for specific issue types with default settings

        .. code-block:: python
            issue_types = {
                "dark": {},
                "blurry": {}
            }
            imagelab.find_issues(issue_types)

        To check for issue types with different hyperparameters.

        .. code-block:: python
            issue_types = {
                "dark": {"threshold": 0.1},
                "blurry": {}
            }
            imagelab.find_issues(issue_types)

        Different issue types can have different hyperparameters

        """
        to_compute_issues_with_params = self._get_issues_to_compute(issue_types)
        print(
            f"Checking for {', '.join([issue_type for issue_type in to_compute_issues_with_params.keys()])} images ..."
        )

        # update issue_types
        self._issue_types = list(
            set(self._issue_types).union(set(to_compute_issues_with_params.keys()))
        )

        # set issue managers
        issue_type_groups = self._get_issue_type_groups(to_compute_issues_with_params)
        self._set_issue_managers(issue_type_groups)

        # find issues
        for issue_type_group, params in issue_type_groups.items():
            issue_manager = self._issue_managers[issue_type_group]
            issue_manager.find_issues(
                params=params, filepaths=self._filepaths, imagelab_info=self.info
            )

            # update issues, issue_summary and info
            self._update_issues(issue_manager.issues)
            self._update_issue_summary(issue_manager.summary)
            self._update_info(issue_manager.info)

        self.issue_summary = self.issue_summary.sort_values(
            by=["num_images"], ascending=False
        )
        self.issue_summary = self.issue_summary.reset_index(drop=True)
        print(
            "Issue checks completed. To see a detailed report of issues found use imagelab.report()."
        )
        return

    def _update_info(self, issue_manager_info: Dict[str, Any]) -> None:
        deep_update_dict(self.info, issue_manager_info)

    def _update_issue_summary(self, issue_manager_summary: pd.DataFrame) -> None:
        # Remove results for issue types computed again
        self.issue_summary = self.issue_summary[
            ~self.issue_summary["issue_type"].isin(issue_manager_summary["issue_type"])
        ]
        # concat new results
        self.issue_summary = pd.concat(
            [self.issue_summary, issue_manager_summary], axis=0, ignore_index=True
        )

    def _update_issues(self, issue_manager_issues: pd.DataFrame) -> None:
        columns_to_update, new_columns = [], []
        for column in issue_manager_issues.columns:
            if column in self.issues.columns:
                columns_to_update.append(column)
            else:
                new_columns.append(column)
        for column_name in columns_to_update:
            self.issues[column_name] = issue_manager_issues[column_name]
        self.issues = self.issues.join(issue_manager_issues[new_columns], how="left")

    def _get_issue_type_groups(
        self, issue_types_with_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        issue_type_groups = {}

        for issue_type, params in issue_types_with_params.items():
            group_name = None
            if issue_type in IMAGE_PROPERTY_ISSUE_TYPES_LIST:
                group_name = IMAGE_PROPERTY
            elif issue_type in DUPLICATE_ISSUE_TYPES_LIST:
                group_name = DUPLICATE
            else:
                issue_type_groups[issue_type] = params

            if group_name:
                if issue_type_groups.get(group_name):
                    issue_type_groups[group_name][issue_type] = params
                else:
                    issue_type_groups[group_name] = {issue_type: params}
        return issue_type_groups

    def _set_issue_managers(self, issue_type_groups: Dict[str, Any]) -> None:
        for issue_type_group, params in issue_type_groups.items():
            self._issue_managers[issue_type_group] = IssueManagerFactory.from_str(
                issue_type_group
            )()

    def _get_topk_issues(self, num_top_issues: int, max_prevalence: float) -> List[str]:
        topk_issues = []
        # Assumes issue_summary is sorted in descending order
        for row in self.issue_summary.itertuples(index=False):
            if getattr(row, "num_images") / self._num_images < max_prevalence:
                topk_issues.append(getattr(row, "issue_type"))
            else:
                print(
                    f"Removing {getattr(row, 'issue_type')} from potential issues in the dataset as it exceeds "
                    f"max_prevalence={max_prevalence} "
                )
        return topk_issues[:num_top_issues]

    def _get_report_args(
        self, verbosity: int, user_supplied_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        report_args = {
            "num_top_issues": self._config["report_num_top_issues_values"][
                verbosity - 1
            ],
            "max_prevalence": self._config["report_max_prevalence"],
            "examples_per_issue": self._config["report_examples_per_issue_values"][
                verbosity - 1
            ],
        }

        non_none_args = {
            arg: value for arg, value in user_supplied_args.items() if value is not None
        }

        return {**report_args, **non_none_args}

    def report(
        self,
        issue_types: Optional[List[str]] = None,
        num_top_issues: Optional[int] = None,
        max_prevalence: Optional[float] = None,
        examples_per_issue: Optional[int] = None,
        verbosity: int = 1,
    ) -> None:
        """Prints a summary of issues found in the dataset with their example images from the dataset.
        By default, if no arguments are specified, it reports the top issues found in the dataset.

        Parameters
        ----------
        issue_types : List[str], optional
            List of issue types to report

        num_top_issues : int, default=3
            Number of top issues to report. It is ignored if issue_types is given

        max_prevalence : float, default=0.5
            Between 0 and 1.
            Ignores the issue types from reporting if found in more than max_prevalence fraction of the dataset.
            It is ignored if issue_types is given.

        examples_per_issue : int, default=4
            Number of examples to show for issue type reported.

        verbosity : int, {1, 2, 3, 4}
            Increasing verbosity increases the detail in the report output

        Examples
        --------
        Default usage

        .. code-block:: python
            imagelab.report()

        Report specific issue types

        .. code-block:: python
            issue_types = ["dark", "near_duplicates"]
            imagelab.report(issue_types=issue_types)

        """
        assert isinstance(verbosity, int) and 0 < verbosity < 5

        user_supplied_args = locals()
        report_args = self._get_report_args(verbosity, user_supplied_args)

        if issue_types:
            computed_issue_types = issue_types
        else:
            print("Top issues in the dataset\n")
            computed_issue_types = self._get_topk_issues(
                report_args["num_top_issues"], report_args["max_prevalence"]
            )
        issue_summary = self.issue_summary[
            self.issue_summary["issue_type"].isin(computed_issue_types)
        ]
        self._pprint_issue_summary(issue_summary)

        self.visualize(
            issue_types=computed_issue_types,
            examples_per_issue=report_args["examples_per_issue"],
        )

    def _pprint_issue_summary(self, issue_summary: pd.DataFrame) -> None:
        issue_summary_copy = issue_summary.copy()
        issue_summary_copy.dropna(axis=1, how="all", inplace=True)
        issue_summary_copy.fillna("N/A", inplace=True)
        print(issue_summary_copy.to_markdown(), "\n")

    def _get_issue_manager(self, issue_type_str: str) -> IssueManager:
        if issue_type_str in IMAGE_PROPERTY_ISSUE_TYPES_LIST:
            return self._issue_managers[IMAGE_PROPERTY]
        elif issue_type_str in DUPLICATE_ISSUE_TYPES_LIST:
            return self._issue_managers[DUPLICATE]
        else:
            return self._issue_managers[issue_type_str]

    def _visualize(
        self,
        issue_type_str: str,
        examples_per_issue: int,
        cell_size: Tuple[int, int],
    ) -> None:
        issue_manager = self._get_issue_manager(issue_type_str)
        viz_name = issue_manager.visualization

        if viz_name == "individual_images":
            sorted_df = self.issues.sort_values(by=[f"{issue_type_str}_score"])
            sorted_df = sorted_df[sorted_df[f"{issue_type_str}_bool"] == 1]
            if len(sorted_df) < examples_per_issue:
                print(
                    f"Found {len(sorted_df)} examples of {issue_type_str} issue in the dataset."
                )
            else:
                print(f"\nTop {examples_per_issue} images with {issue_type_str} issue")

            sorted_filepaths = sorted_df.index[:examples_per_issue].tolist()
            if sorted_filepaths:
                VizManager.individual_images(
                    filepaths=sorted_filepaths,
                    ncols=self._config["visualize_num_images_per_row"],
                    cell_size=cell_size,
                    cmap="gray"
                    if issue_type_str == IssueType.GRAYSCALE.value
                    else None,
                )
        elif viz_name == "image_sets":
            image_sets = self.info[issue_type_str][SETS][:examples_per_issue]
            if len(image_sets) < examples_per_issue:
                print(
                    f"Found {len(image_sets)} sets of images with {issue_type_str} issue in the dataset."
                )
            else:
                print(
                    f"\nTop {examples_per_issue} sets of images with {issue_type_str} issue"
                )
            if image_sets:
                VizManager.image_sets(
                    image_sets,
                    ncols=self._config["visualize_num_images_per_row"],
                    cell_size=cell_size,
                )

    def visualize(
        self,
        image_files: Optional[List[str]] = None,
        issue_types: Optional[List[str]] = None,
        num_images: int = 4,
        examples_per_issue: int = 4,
        cell_size: Tuple[int, int] = (2, 2),
    ) -> None:
        """Visualization helper for images.

        This method is multipurpose and can be used for visualizaing:
        1. Random images from the dataset
        2. Particular images with paths given in image_files
        3. Top examples of given issue_types found in the dataset

        If no image_files or issue_types are given, random images will be shown from the dataset.
        If specific image_files are given, it will override the argument issue_types and will show given image_files
        If image_files are not given and issue_types are given, top examples of given image_types will be shown

        Parameters
        ----------

        image_files : List[str], optional
            List of image filepaths to visualize

        issue_types: List[str], optional
            List of issue types to visualize

        num_images : int, optional
            Number of images to randomly visualize from the dataset
            Used only when image_files and issue_types are empty, otherwise this argument will be ignored

        examples_per_issue : int, optional
            Number of top examples per issue type to visualize
            Used only when issue_types is given and image_files is empty

        cell_size : Tuple[int, int], optional
            Dimensions controlling the size of each image in the depicted image grid.

        Examples
        --------

        To visualize random images from the dataset

        .. code-block:: python
            imagelab.visualize()

        .. code-block:: python
            imagelab.visualize(num_images=8)

        To visualize specfic images from the dataset

        .. code-block:: python
            image_files = ["./dataset/cat.png", "./dataset/dog.png", "./dataset/mouse.png"]
            imagelab.visualize(image_files=image_files)


        To visualize top examples of specific issue types from the dataset

        .. code-block:: python
            issue_types = ["dark", "odd_aspect_ratio"]
            imagelab.visualize(issue_types=issue_types)

        """
        if issue_types:
            for issue_type in issue_types:
                self._visualize(issue_type, examples_per_issue, cell_size)
        else:
            if not image_files:
                image_files = list(
                    np.random.choice(self._filepaths, num_images, replace=False)
                )
            VizManager.individual_images(
                filepaths=image_files,
                ncols=self._config["visualize_num_images_per_row"],
                cell_size=cell_size,
            )

    # Todo: Improve mypy dict typechecking so this does not return any
    def get_stats(self) -> Any:
        return self.info["statistics"]

    def save(self, path: str) -> None:
        """Saves this ImageLab instance into a folder at the given path.
        Your saved Imagelab should be loaded from the same version of the CleanVision package.
        This method does not save your image files.

        Parameters
        ----------
        path : str
            path at which to save the Imagelab instance
        """
        if os.path.exists(path):
            print(
                f"WARNING: Existing files will be overwritten by newly saved files at: {path}"
            )
        else:
            os.mkdir(path)

        self._path = path
        object_file = os.path.join(self._path, OBJECT_FILENAME)
        with open(object_file, "wb") as f:
            pickle.dump(self, f)

        print(f"Saved Imagelab to folder: {path}")
        print(
            "The data path and dataset must be not be changed to maintain consistent state when loading this Imagelab"
        )

    @classmethod
    def load(
        cls: Type[TImagelab], path: str, data_path: Optional[str] = None
    ) -> TImagelab:
        """Loads Imagelab from given path.


        Parameters
        ----------
        path : str
            Path to the saved Imagelab, not pickle file.
        data_path : str
            Path to image dataset previously used in Imagelab.
            If the `data_path` is changed, Imagelab will not be loaded as some of its functionalities depend on it.
            You should be using the same version of the CleanVision package previously used when saving the Imagelab.

        Returns
        -------
        Imagelab
            Returns a saved instance of Imagelab
        """
        if not os.path.exists(path):
            raise ValueError(f"No folder found at specified path: {path}")

        object_file = os.path.join(path, OBJECT_FILENAME)
        with open(object_file, "rb") as f:
            imagelab: TImagelab = pickle.load(f)

        if data_path is not None:
            filepaths = get_filepaths(data_path)
            if set(filepaths) != set(imagelab._filepaths):
                raise ValueError(
                    "Absolute path of image(s) has changed in the dataset. Cannot load Imagelab."
                )
        print("Successfully loaded Imagelab")
        return imagelab
