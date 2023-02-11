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

OBJECT_FILENAME = "imagelab.pkl"
TImagelab = TypeVar("TImagelab", bound="Imagelab")


class Imagelab:
    def __init__(self, data_path: str) -> None:
        self.filepaths: List[str] = get_filepaths(data_path)
        self.num_images: int = len(self.filepaths)
        if self.num_images == 0:
            raise ValueError(f"No images found in the specified path:{data_path}")
        self.info: Dict[str, Any] = {"statistics": {}}
        self.issue_summary: pd.DataFrame = pd.DataFrame(columns=["issue_type"])
        self.issues: pd.DataFrame = pd.DataFrame(index=self.filepaths)
        self.issue_types: List[str] = []
        self.issue_managers: Dict[str, IssueManager] = {}
        # can be loaded from a file later
        self.config: Dict[str, Any] = self._set_default_config()
        self.path = ""

    def _set_default_config(self) -> Dict[str, Any]:
        return {
            "visualize_num_images_per_row": 4,
            "report_num_top_issues_values": [3, 5, 10, len(self.issue_types)],
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
        print("Default issue type checked by Imagelab:\n")
        print(
            *[issue_type.value for issue_type in self.config["default_issue_types"]],
            sep="\n",
        )

    def list_possible_issue_types(self) -> None:
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
                for issue_type in self.config["default_issue_types"]
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
        to_compute_issues_with_params = self._get_issues_to_compute(issue_types)
        print(
            f"Checking for {', '.join([issue_type for issue_type in to_compute_issues_with_params.keys()])} images ..."
        )

        # update issue_types
        self.issue_types = list(
            set(self.issue_types).union(set(to_compute_issues_with_params.keys()))
        )

        # set issue managers
        issue_type_groups = self._get_issue_type_groups(to_compute_issues_with_params)
        self._set_issue_managers(issue_type_groups)

        # find issues
        for issue_type_group, params in issue_type_groups.items():
            issue_manager = self.issue_managers[issue_type_group]
            issue_manager.find_issues(
                params=params,
                filepaths=self.filepaths,
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
            self.issue_managers[issue_type_group] = IssueManagerFactory.from_str(
                issue_type_group
            )()

    def _get_topk_issues(self, num_top_issues: int, max_prevalence: float) -> List[str]:
        topk_issues = []
        # Assumes issue_summary is sorted in descending order
        for row in self.issue_summary.itertuples(index=False):
            if getattr(row, "num_images") / self.num_images < max_prevalence:
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
            "num_top_issues": self.config["report_num_top_issues_values"][
                verbosity - 1
            ],
            "max_prevalence": self.config["report_max_prevalence"],
            "examples_per_issue": self.config["report_examples_per_issue_values"][
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
        self.print_issue_summary(issue_summary)

        self.visualize(
            issue_types=computed_issue_types,
            examples_per_issue=report_args["examples_per_issue"],
        )

    def print_issue_summary(self, issue_summary: pd.DataFrame) -> None:
        issue_summary_copy = issue_summary.copy()
        issue_summary_copy.dropna(axis=1, how="all", inplace=True)
        issue_summary_copy.fillna("N/A", inplace=True)
        print(issue_summary_copy.to_markdown(), "\n")

    def _get_issue_manager(self, issue_type_str: str) -> IssueManager:
        if issue_type_str in IMAGE_PROPERTY_ISSUE_TYPES_LIST:
            return self.issue_managers[IMAGE_PROPERTY]
        elif issue_type_str in DUPLICATE_ISSUE_TYPES_LIST:
            return self.issue_managers[DUPLICATE]
        else:
            return self.issue_managers[issue_type_str]

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
                    ncols=self.config["visualize_num_images_per_row"],
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
                    ncols=self.config["visualize_num_images_per_row"],
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
        if issue_types:
            for issue_type in issue_types:
                self._visualize(issue_type, examples_per_issue, cell_size)
        else:
            if not image_files:
                image_files = list(
                    np.random.choice(self.filepaths, num_images, replace=False)
                )
            VizManager.individual_images(
                filepaths=image_files,
                ncols=self.config["visualize_num_images_per_row"],
                cell_size=cell_size,
            )

    # Todo: Improve mypy dict typechecking so this does not return any
    def get_stats(self) -> Any:
        return self.info["statistics"]

    def save(self, path: str) -> None:
        """Saves this ImageLab to file (all files are in folder at path/).
        Your saved Imagelab should be loaded from the same version of the CleanVision package.
        This method does not save your image files.
        """
        if os.path.exists(path):
            print(
                f"WARNING: Existing files will be overwritten by newly saved files at: {path}"
            )
        else:
            os.mkdir(path)

        self.path = path
        object_file = os.path.join(self.path, OBJECT_FILENAME)
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
        """Loads Imagelab from file.
        `path` is the path to the saved Imagelab, not pickle file.
        `data_path` is the path to image dataset previously used in Imagelab.
        If the `data_path` is changed, Imagelab will not be loaded as some of its functionalities depend on it.
        You should be using the same version of the CleanVision package previously used when saving the Imagelab.
        """
        if not os.path.exists(path):
            raise ValueError(f"No folder found at specified path: {path}")

        object_file = os.path.join(path, OBJECT_FILENAME)
        with open(object_file, "rb") as f:
            imagelab: TImagelab = pickle.load(f)

        if data_path is not None:
            filepaths = get_filepaths(data_path)
            if set(filepaths) != set(imagelab.filepaths):
                raise ValueError(
                    "Absolute path of image(s) has changed in the dataset. Cannot load Imagelab."
                )
        print("Successfully loaded Imagelab")
        return imagelab
