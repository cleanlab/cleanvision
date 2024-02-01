"""
Imagelab is the core class in CleanVision for finding all types of issues in an image dataset.
The methods in this module should suffice for most use-cases,
but advanced users can get extra flexibility via the code in other CleanVision modules.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np
import pandas as pd
from PIL import Image

import cleanvision
from cleanvision.dataset.torch_dataset import TorchDataset
from cleanvision.dataset.utils import build_dataset
from cleanvision.issue_managers import (
    ISSUE_MANAGER_REGISTRY,
    IssueManagerFactory,
    IssueType,
)
from cleanvision.utils.base_issue_manager import IssueManager
from cleanvision.utils.constants import (
    DEFAULT_ISSUE_TYPES,
    DUPLICATE,
    DUPLICATE_ISSUE_TYPES_LIST,
    IMAGE_PROPERTY,
    IMAGE_PROPERTY_ISSUE_TYPES_LIST,
    SETS,
)
from cleanvision.utils.serialize import Serializer
from cleanvision.utils.utils import (
    deep_update_dict,
    get_is_issue_colname,
    get_max_n_jobs,
    get_score_colname,
    update_df,
)
from cleanvision.utils.viz_manager import VizManager

if TYPE_CHECKING:  # pragma: no cover
    import datasets
    from torchvision.datasets.vision import VisionDataset

__all__ = ["Imagelab"]
TImagelab = TypeVar("TImagelab", bound="Imagelab")


class Imagelab:
    """A single class to find all types of issues in image datasets.
    Imagelab detects issues in any image dataset and thus can be useful in most computer vision tasks including
    supervised and unsupervised training.
    Imagelab supports various formats for datasets: local folder containing images, a list of image
    filepaths, HuggingFace dataset and Torchvision dataset.
    Specify only one of these arguments: `data_path`, `filepaths`, (`hf_dataset`, `image_key`), `torchvision_dataset`



    Parameters
    ----------
    data_path : str
        Path to image files.
        Imagelab will recursively retrieve all image files from the specified path

    filepaths: List[str], optional
        Issue checks will be run on this list of image paths specified in `filepaths`.

    hf_dataset: datasets.Dataset
        Hugging Face dataset with images in PIL format accessible via some key in ``hf_dataset.features``.

    image_key: str
        Key used to access images within the Hugging Face `dataset.features` object. For many datasets, this key is just called "image".
        This argument must be specified if you provide a Hugging Face dataset; for other types of dataset this argument has no effect.

    torchvision_dataset: torchvision.datasets.vision.VisionDataset
        torchvision dataset where each individual  example is a tuple containing exactly one image in PIL format.

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

        from cleanvision import Imagelab
        imagelab = Imagelab(data_path="FOLDER_WITH_IMAGES/")
        imagelab.find_issues()
        imagelab.report()

    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        filepaths: Optional[List[str]] = None,
        hf_dataset: Optional["datasets.Dataset"] = None,
        image_key: Optional[str] = None,
        torchvision_dataset: Optional["VisionDataset"] = None,
        storage_opts: Dict[str, Any] = {},
    ) -> None:
        self._dataset = build_dataset(
            data_path,
            filepaths,
            hf_dataset,
            image_key,
            torchvision_dataset,
            storage_opts=storage_opts,
        )
        if len(self._dataset) == 0:
            raise ValueError("No images found in the dataset specified")
        self.info: Dict[str, Any] = {"statistics": {}}
        self.issue_summary: pd.DataFrame = pd.DataFrame(
            columns=["issue_type", "num_images"]
        ).astype({"issue_type": str, "num_images": np.int64})

        self.issues: pd.DataFrame = pd.DataFrame(index=self._dataset.index)
        self._issue_types: List[str] = []
        self._issue_managers: Dict[str, IssueManager] = {}

        # TODO: can be loaded from a file later
        self._config: Dict[str, Any] = self._set_default_config()
        self.cleanvision_version: str = cleanvision.__version__

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
            "report_num_images": 4,
            "report_max_prevalence": 0.5,
            "report_cell_size": (2, 2),
        }

    @staticmethod
    def list_default_issue_types() -> List[str]:
        """Returns a list of the issue types that are run by default in :py:meth:`Imagelab.find_issues`"""
        return DEFAULT_ISSUE_TYPES

    @staticmethod
    def list_possible_issue_types() -> List[str]:
        """Returns a list of all the possible issue types that can be run in :py:meth:`Imagelab.find_issues`
        This list will also include custom issue types if properly added.
        """
        issue_types = Imagelab.list_default_issue_types()
        for key in ISSUE_MANAGER_REGISTRY:
            if key not in [IMAGE_PROPERTY, DUPLICATE]:
                issue_types.append(key)
        return list(set(issue_types))

    def _get_issues_to_compute(
        self, issue_types_with_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not issue_types_with_params:
            to_compute_issues_with_params: Dict[str, Any] = {
                issue_type: {} for issue_type in self.list_default_issue_types()
            }
        else:
            to_compute_issues_with_params = {
                issue_type_str: params
                for issue_type_str, params in issue_types_with_params.items()
            }
        return to_compute_issues_with_params

    def find_issues(
        self,
        issue_types: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
        verbose: bool = True,
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
        verbose : bool, default=True
            If True, prints helpful information while checking for issues.

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
        if verbose:
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

        # set number of jobs for parallelizing computation
        if n_jobs is None:
            n_jobs = get_max_n_jobs()
            if isinstance(self._dataset, TorchDataset):
                n_jobs = 1

        # find issues
        for issue_type_group, params in issue_type_groups.items():
            issue_manager = self._issue_managers[issue_type_group]
            issue_manager.find_issues(
                params=params,
                dataset=self._dataset,
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
        if verbose:
            print(
                f"Issue checks completed. {self.issue_summary['num_images'].sum()} issues found in the dataset. To see a detailed report of issues found, use imagelab.report()."
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
        """Filters issues by max_prevalence in the dataset."""
        issue_summary = self.issue_summary[
            self.issue_summary["issue_type"].isin(issue_types)
        ]
        issue_to_report = []
        for row in issue_summary.itertuples(index=False):
            if getattr(row, "num_images") / len(self._dataset) < max_prevalence:
                issue_to_report.append(getattr(row, "issue_type"))
            else:
                print(
                    f"Removing {getattr(row, 'issue_type')} from potential issues in the dataset as it exceeds "
                    f"max_prevalence={max_prevalence} "
                )
        return issue_to_report

    def _get_report_args(self, user_supplied_args: Dict[str, Any]) -> Dict[str, Any]:
        report_args = {
            "max_prevalence": self._config["report_max_prevalence"],
            "num_images": self._config["report_num_images"],
            "cell_size": self._config["report_cell_size"],
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
        print_summary: bool = True,
        show_id: bool = False,
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
            Increasing verbosity increases the detail of the report. Set this to 1 to report less information, or to 4 to report the most information.

        print_summary : bool, default=True
            If True, prints the summary of issues found in the dataset.

        show_id: bool, default=False
            If True, prints the dataset ID of each image shown in the report.

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
        assert isinstance(verbosity, int) and 0 <= verbosity < 5

        user_supplied_args = locals()
        report_args = self._get_report_args(user_supplied_args)

        issue_types_to_report = (
            issue_types if issue_types else self.issue_summary["issue_type"].tolist()
        )

        # filter issues based on max_prevalence in the dataset
        filtered_issue_types = self._filter_report(
            issue_types_to_report, report_args["max_prevalence"]
        )

        issue_summary = self.issue_summary[
            self.issue_summary["issue_type"].isin(filtered_issue_types)
        ]
        if len(issue_summary) > 0:
            if verbosity:
                print("Issues found in images in order of severity in the dataset\n")
            if print_summary:
                self._pprint_issue_summary(issue_summary)
            for issue_type in filtered_issue_types:
                if (
                    self.issue_summary.query(f"issue_type == '{issue_type}'")[
                        "num_images"
                    ].values[0]
                    == 0
                ):
                    continue
                print(f"{' ' + issue_type + ' images ':-^60}\n")
                print(
                    f"Number of examples with this issue: {self.issues[get_is_issue_colname(issue_type)].sum()}\n"
                    f"Examples representing most severe instances of this issue:\n"
                )
                self._visualize(
                    issue_type,
                    report_args["num_images"],
                    report_args["cell_size"],
                    show_id,
                )
        else:
            print(
                "Please specify some issue_types to check for in imagelab.find_issues()."
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
        issue_type: str,
        num_images: int,
        cell_size: Tuple[int, int],
        show_id: bool,
    ) -> None:
        # todo: remove dependency on issue manager
        issue_manager = self._get_issue_manager(issue_type)
        viz_name = issue_manager.visualization

        if viz_name == "individual_images":
            sorted_df = self.issues.sort_values(by=get_score_colname(issue_type))
            sorted_df = sorted_df[sorted_df[get_is_issue_colname(issue_type)] == 1]

            scores = sorted_df.head(num_images)[get_score_colname(issue_type)]
            indices = scores.index.tolist()
            images = [self._dataset[i] for i in indices]

            # construct title info
            title_info = {"scores": [f"score : {x:.4f}" for x in scores]}
            if show_id:
                title_info["ids"] = [f"id : {i}" for i in indices]
            if issue_type == IssueType.ODD_SIZE.value:
                title_info["size"] = [f"size: {image.size}" for image in images]

            if images:
                VizManager.individual_images(
                    images=images,
                    title_info=title_info,
                    ncols=self._config["visualize_num_images_per_row"],
                    cell_size=cell_size,
                )

        elif viz_name == "image_sets":
            image_sets_indices = sorted(
                self.info[issue_type][SETS], key=len, reverse=True
            )
            image_sets_indices = image_sets_indices[:num_images]
            image_sets = []
            for indices in image_sets_indices:
                image_sets.append([self._dataset[index] for index in indices])

            title_info_sets = []
            for s in image_sets_indices:
                title_info = {"name": [self._dataset.get_name(index) for index in s]}
                title_info_sets.append(title_info)

            if image_sets:
                VizManager.image_sets(
                    image_sets,
                    title_info_sets,
                    ncols=self._config["visualize_num_images_per_row"],
                    cell_size=cell_size,
                )

    # todo: compress this code
    def visualize(
        self,
        image_files: Optional[List[str]] = None,
        indices: Optional[List[str | int]] = None,
        issue_types: Optional[List[str]] = None,
        num_images: int = 4,
        cell_size: Tuple[int, int] = (2, 2),
        show_id: bool = False,
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

        indices: List[str|int], optional
            List of indices of images in the dataset to visualize.
            If the dataset is a local data_path, the indices are filepaths, which is also the index in `imagelab.issues` dataframe.
            If the dataset is a huggingface or torchvision dataset, indices are of type int and corresponding to the indices in the dataset object.


        issue_types: List[str], optional
            List of issue types to visualize. For each type of issue, will show a few images representing the top-most severe instances of this issue in the dataset.

        num_images : int, optional
            Number of images to visualize from the dataset.
            These images are randomly selected if `issue_types` is ``None``.
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
        if issue_types is not None:
            if len(issue_types) == 0:
                raise ValueError("issue_types list is empty")
            for issue_type in issue_types:
                self._visualize(issue_type, num_images, cell_size, show_id)
        elif image_files is not None:
            if len(image_files) == 0:
                raise ValueError("image_files list is empty.")
            images = [Image.open(path) for path in image_files]
            title_info = {"path": [path.split("/")[-1] for path in image_files]}
            VizManager.individual_images(
                images,
                title_info,
                ncols=self._config["visualize_num_images_per_row"],
                cell_size=cell_size,
            )
        elif indices:
            images = [self._dataset[i] for i in indices]
            title_info = {"name": [self._dataset.get_name(i) for i in indices]}
            VizManager.individual_images(
                images,
                title_info,
                ncols=self._config["visualize_num_images_per_row"],
                cell_size=cell_size,
            )
        else:
            print("Sample images from the dataset")

            if image_files is None:
                image_indices = random.sample(
                    self._dataset.index, min(num_images, len(self._dataset))
                )
                images = [self._dataset[i] for i in image_indices]
                title_info = {
                    "name": [self._dataset.get_name(i) for i in image_indices]
                }
                VizManager.individual_images(
                    images,
                    title_info,
                    ncols=self._config["visualize_num_images_per_row"],
                    cell_size=cell_size,
                )

    # Todo: Improve mypy dict typechecking so this does not return any
    def get_stats(self) -> Any:
        """Returns dict of statistics computed from images when auditing the data such as: brightness, color space, aspect ratio, etc.
        If statistics have not been computed yet, then returns ``None``.
        """
        return self.info["statistics"]

    def save(self, path: str, force: bool = False) -> None:
        """Saves this Imagelab instance, :py:attr:`issues` and :py:attr:`issue_summary` into a folder at the given path.
        Your saved Imagelab should be loaded from the same version of the CleanVision package to avoid inconsistencies.
        This method does not save your image files.

        Parameters
        ----------
        path : str
            Path to folder where this Imagelab instance will be saved on disk.

        force: bool, default=False
            If set to True, any existing files at `path` will be overwritten.
        """
        Serializer.serialize(path=path, imagelab=self, force=force)

    @classmethod
    def load(
        cls: Type[TImagelab], path: str, data_path: Optional[str] = None
    ) -> Imagelab:
        """Loads Imagelab from given path.

        Parameters
        ----------
        path : str
            Path to the saved Imagelab folder previously specified in :py:meth:`Imagelab.save` (not the individual pickle file).
        data_path : str
            Path to image dataset previously used in Imagelab, if your data exists locally as images in a folder.
            If the `data_path` is changed, the code will break as Imagelab functionalities are dependent on it.
            You should be using the same version of the CleanVision package previously used when saving Imagelab.

        Returns
        -------
        Imagelab
            Returns a saved instance of Imagelab
        """
        imagelab = Serializer.deserialize(path)
        return imagelab
