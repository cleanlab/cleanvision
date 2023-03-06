"""
Imagelab is the core class in CleanVision for finding all types of issues in an image dataset.
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
from cleanvision.utils.utils import (
    get_filepaths,
    deep_update_dict,
    get_is_issue_colname,
    get_score_colname,
    update_df,
)
from cleanvision.utils.viz_manager import VizManager

__all__ = ["Imagelab"]

OBJECT_FILENAME = "imagelab.pkl"
TImagelab = TypeVar("TImagelab", bound="Imagelab")


class Imagelab:
    """A single class to find all types of issues in image datasets. Imagelab detects issues in the raw image files themselves and thus can be useful in most computer vision tasks.

    Parameters
    ----------
    data_path : str
        Path to image files.
        Imagelab will recursively retrieve all image files from the specified path

    filepaths: List[str], optional
        Issue checks will be run on this list of image paths specified in `filepaths`.
        Specifying only one of `data_path` or `filepaths`.

    Attributes
    ----------
    issues : pd.DataFrame
        Dataframe where each row corresponds to an image and columns specify which issues were detected in this image.
        It has two types of columns for each issue type:

        1. <issue_type>_score - This column contains a quality-score for each image for a particular type of issue.
        Scores are between 0 and 1, lower values indicate images exhibiting more severe instances of this issue.

        2. is_<issue_type>_issue - This column indicates whether or not the issue_type is detected in each image (a binary decision rather than numeric score).

    issue_summary : pd.DataFrame
        Dataframe where each row corresponds to a type of issue and columns summarize the overall prevalence of this issue in the dataset.
        Specifically, it shows the number of images detected with the issue.

    info : Dict
        Nested dictionary that contains statistics and other useful information about the dataset.
        Also contains additional information saved while checking for issues in the dataset.

    Raises
    ------
    ValueError
        If no images are found in the specified paths.
        If both `data_path` and `filepaths` are given or none of them are specified.

    Examples
    --------

    Basic usage of Imagelab class

    .. code-block:: python

        from cleanvision.imagelab import Imagelab
        imagelab = Imagelab(data_path="FOLDER_WITH_IMAGES/")
        imagelab.find_issues()
        imagelab.report()

    """

    def __init__(
        self, data_path: Optional[str] = None, filepaths: Optional[List[str]] = None
    ) -> None:
        self._filepaths = self._get_filepaths(data_path, filepaths)
        self._num_images: int = len(self._filepaths)
        if self._num_images == 0:
            raise ValueError("No images found in the specified path")
        self.info: Dict[str, Any] = {"statistics": {}}
        self.issue_summary: pd.DataFrame = pd.DataFrame(columns=["issue_type"])
        self.issues: pd.DataFrame = pd.DataFrame(index=self._filepaths)
        self._issue_types: List[str] = []
        self._issue_managers: Dict[str, IssueManager] = {}
        # can be loaded from a file later
        self._config: Dict[str, Any] = self._set_default_config()
        self._path = ""

    def _get_filepaths(
        self, data_path: Optional[str], filepaths: Optional[List[str]]
    ) -> List[str]:
        if (data_path is None and filepaths is None) or (
            data_path is not None and filepaths is not None
        ):
            raise ValueError(
                "Please specify one of data_path or filepaths to check for issues."
            )
        filepaths = []
        if data_path:
            filepaths = get_filepaths(data_path)
        elif filepaths:
            filepaths = filepaths
        return filepaths

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
            "report_num_top_issues_values": [5, 7, 10, len(self._issue_types)],
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
        """Prints list of the issue types detected by default if no types are specified in :py:meth:`Imagelab.find_issues`"""

        print("Default issue type checked by Imagelab:\n")
        print(
            *[issue_type.value for issue_type in self._config["default_issue_types"]],
            sep="\n",
        )

    def list_possible_issue_types(self) -> None:
        """Prints list of all possible issue types that can be detected in a dataset.
        This list will also include custom issue types if you properly add them.
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
        self, issue_types: Optional[Dict[str, Any]] = None, n_jobs: Optional[int] = None
    ) -> None:
        """Finds issues in the dataset.
        If `issue_types` is not provided, dataset is checked for a default set of issue types.
        To see default set: :py:meth:`Imagelab.list_default_issue_types`

        Parameters
        ----------
        issue_types : Dict[str, Any], optional
            Dict with issue types to check as keys.
            The value of this dict is a dict containing hyperparameters for each issue type.
        n_jobs :  int, default=None
            Number of processing threads used by multiprocessing.
            Default None sets to the number of cores on your CPU (physical cores if you have psutil package installed, otherwise logical cores).
            Set this to 1 to disable parallel processing (if its causing issues). Windows users may see a speed-up with n_jobs=1.

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

        To check for issue types with different hyperparameters. Different issue types can have different hyperparameters.

        .. code-block:: python

            issue_types = {
                "dark": {"threshold": 0.1},
                "blurry": {}
            }
            imagelab.find_issues(issue_types)



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
                params=params,
                filepaths=self._filepaths,
                imagelab_info=self.info,
                n_jobs=n_jobs,
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
            "Issue checks completed. To see a detailed report of issues found, use imagelab.report()."
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
        self.issues = update_df(self.issues, issue_manager_issues)

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

    def _filter_report(
        self, issue_types: List[str], max_prevalence: float
    ) -> List[str]:
        issue_summary = self.issue_summary[
            self.issue_summary["issue_type"].isin(issue_types)
        ]
        issue_to_report = []
        for row in issue_summary.itertuples(index=False):
            if getattr(row, "num_images") / self._num_images < max_prevalence:
                issue_to_report.append(getattr(row, "issue_type"))
            else:
                print(
                    f"Removing {getattr(row, 'issue_type')} from potential issues in the dataset as it exceeds "
                    f"max_prevalence={max_prevalence} "
                )
        return issue_to_report

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
        max_prevalence: Optional[float] = None,
        num_images: Optional[int] = None,
        verbosity: int = 1,
    ) -> None:
        """Prints summary of the issues found in your dataset.
        By default, this method depicts the images representing top-most severe instances of each issue type.

        Parameters
        ----------
        issue_types : List[str], optional
            List of issue types to consider in report.
            This must be subset of the issue types specified in :py:meth:`Imagelab.find_issues``.

        max_prevalence : float, default=0.5
            Value between 0 and 1
            Issue types that are detected in more than `max_prevalence` fraction of the images in dataset will be omitted from the report.
            You are presumably already aware of these in your dataset.

        num_images : int, default=4
            Maximum number of images to show for issue type reported. These are examples of the top-most severe instances of the issue in your dataset.

        verbosity : int, {1, 2, 3, 4}
            Increasing verbosity increases the detail of the report. Set this to 0 to report less information, or to 4 to report the most information.

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
            issue_types_to_report = issue_types
        else:
            non_zero_issue_types = self.issue_summary[
                self.issue_summary["num_images"] > 0
            ]["issue_type"].tolist()
            issue_types_to_report = non_zero_issue_types

        filtered_issue_types = self._filter_report(
            issue_types_to_report, report_args["max_prevalence"]
        )

        num_report = (
            report_args["num_top_issues"] if issue_types is None else len(issue_types)
        )

        issue_summary = self.issue_summary[
            self.issue_summary["issue_type"].isin(filtered_issue_types[:num_report])
        ]
        if len(issue_summary) > 0:
            print("Issues found in order of severity in the dataset\n")
            self._pprint_issue_summary(issue_summary)
            if issue_types is None and num_report < len(filtered_issue_types):
                print(
                    f"{len(filtered_issue_types) - num_report} more issues found in the dataset. To view them increase verbosity or check imagelab.issue_summary."
                )

            self.visualize(
                issue_types=filtered_issue_types[:num_report],
                num_images=(
                    report_args["examples_per_issue"]
                    if num_images is None
                    else num_images
                ),
            )
        else:
            print("No issues found.")

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
        issue_type: str,
        num_images: int,
        cell_size: Tuple[int, int],
    ) -> None:
        # todo: remove dependency on issue manager
        issue_manager = self._get_issue_manager(issue_type)
        viz_name = issue_manager.visualization

        if viz_name == "individual_images":
            sorted_df = self.issues.sort_values(by=get_score_colname(issue_type))
            sorted_df = sorted_df[sorted_df[get_is_issue_colname(issue_type)] == 1]

            examples_str = "examples" if len(sorted_df) > 1 else "example"
            if len(sorted_df) < num_images:
                print(
                    f"Found {len(sorted_df)} {examples_str} with {issue_type} issue in the dataset."
                )
            else:
                print(
                    f"\nTop {num_images} {examples_str} with {issue_type} issue in the dataset."
                )

            scores = sorted_df.head(num_images)[get_score_colname(issue_type)]
            titles = [f"score : {x:.4f}" for x in scores]
            paths = scores.index.tolist()
            if paths:
                VizManager.individual_images(
                    filepaths=paths,
                    titles=titles,
                    ncols=self._config["visualize_num_images_per_row"],
                    cell_size=cell_size,
                )

        elif viz_name == "image_sets":
            image_sets = list(self.info[issue_type][SETS][:num_images])

            sets_str = "sets" if len(image_sets) > 1 else "set"
            if len(image_sets) < num_images:
                print(
                    f"Found {len(image_sets)} {sets_str} of images with {issue_type} issue in the dataset."
                )
            else:
                print(
                    f"\nTop {num_images} {sets_str} of images with {issue_type} issue"
                )

            title_sets = [[path.split("/")[-1] for path in s] for s in image_sets]

            if image_sets:
                VizManager.image_sets(
                    image_sets,
                    title_sets,
                    ncols=self._config["visualize_num_images_per_row"],
                    cell_size=cell_size,
                )

    def visualize(
        self,
        image_files: Optional[List[str]] = None,
        issue_types: Optional[List[str]] = None,
        num_images: int = 4,
        cell_size: Tuple[int, int] = (2, 2),
    ) -> None:
        """Show specific images.

        Can be used for visualizing either:
        1. Particular images with paths given in `image_files`.
        2. Images representing top-most severe instances of given `issue_types` detected the dataset.
        3. If no `image_files` or `issue_types` are given, random images will be shown from the dataset.

        If `image_files` is given, this overrides the argument `issue_types`.

        Parameters
        ----------

        image_files : List[str], optional
            List of filepaths for images to visualize.

        issue_types: List[str], optional
            List of issue types to visualize. For each type of issue, will show a few images representing the top-most severe instances of this issue in the dataset.

        num_images : int, optional
            Number of images to visualize from the dataset.
            These are randomly selected if `issue_types` is ``None``.
            If `issue_types` is given, then this is the number of images for each issue type to visualize
            (images representing top-most severe instances of this issue will be shown).
            If `image_files` is given, this argument is ignored.

        cell_size : Tuple[int, int], optional
            Dimensions controlling the size of each image in the depicted image grid.

        Examples
        --------

        To visualize random images from the dataset

        .. code-block:: python

            imagelab.visualize()

        .. code-block:: python

            imagelab.visualize(num_images=8)

        To visualize specific images from the dataset

        .. code-block:: python

            image_files = ["./dataset/cat.png", "./dataset/dog.png", "./dataset/mouse.png"]
            imagelab.visualize(image_files=image_files)

        To visualize top examples of specific issue types from the dataset

        .. code-block:: python

            issue_types = ["dark", "odd_aspect_ratio"]
            imagelab.visualize(issue_types=issue_types)

        """
        if issue_types:
            if len(issue_types) == 0:
                raise ValueError("issue_types list is empty")
            for issue_type in issue_types:
                self._visualize(issue_type, num_images, cell_size)
        else:
            if image_files is None:
                image_files = list(
                    np.random.choice(
                        self._filepaths,
                        min(num_images, self._num_images),
                        replace=False,
                    )
                )
            elif len(image_files) == 0:
                raise ValueError("image_files list is empty.")
            if image_files:
                titles = [path.split("/")[-1] for path in image_files]
                VizManager.individual_images(
                    image_files,
                    titles,
                    ncols=self._config["visualize_num_images_per_row"],
                    cell_size=cell_size,
                )

    # Todo: Improve mypy dict typechecking so this does not return any
    def get_stats(self) -> Any:
        return self.info["statistics"]

    def save(self, path: str, force: bool = False) -> None:
        """Saves this ImageLab instance into a folder at the given path.
        Your saved Imagelab should be loaded from the same version of the CleanVision package.
        This method does not save your image files.

        Parameters
        ----------
        path : str
            Path to folder where this Imagelab instance will be saved on disk.

        allow_overwrite: bool, default=False
            If set to True, any existing files at `path` will be overwritten.

        Raises
        ------
        ValueError
            If `allow_overwrite` is set to False, and an existing path is specified for saving Imagelab instance.
        """
        path_exists = os.path.exists(path)
        if not path_exists:
            os.mkdir(path)
        else:
            if force:
                print(
                    f"WARNING: Existing files will be overwritten by newly saved files at: {path}"
                )
            else:
                raise FileExistsError("Please specify a new path or set force=True")

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
            Path to the saved Imagelab folder previously specified in :py:meth:`Imagelab.save` (not the individual pickle file).
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
