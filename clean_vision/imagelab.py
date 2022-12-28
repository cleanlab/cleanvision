import math

import pandas as pd

from clean_vision.constants import IMAGE_PROPERTY
from clean_vision.issue_types import IssueType
from clean_vision.utils.issue_manager_factory import _IssueManagerFactory
from clean_vision.utils.utils import get_filepaths
from clean_vision.viz_manager import VizManager


class Imagelab:
    def __init__(self, data_path):
        self.filepaths = get_filepaths(data_path)
        self.num_images = len(self.filepaths)
        self.info = {}
        self.issue_summary = pd.DataFrame()
        # self.issues = pd.DataFrame(self.filepaths, columns=["image_path"])
        self.issues = pd.DataFrame(index=self.filepaths)
        self.issue_types = []
        self.issue_managers = {}
        # can be loaded from a file later
        self.config = {"viz_num_images_per_row": 4}

    def _get_issues_to_compute(self, issue_types):
        if issue_types is None or len(issue_types) == 0:
            all_issues = list(IssueType)
        else:
            all_issues = []
            for issue_type_str, threshold in issue_types.items():
                issue_type = IssueType(issue_type_str)
                issue_type.threshold = threshold
                all_issues.append(issue_type)
        to_compute_issues = list(set(all_issues) - set(self.issue_types))
        return to_compute_issues

    def find_issues(self, issue_types=None):
        to_compute_issues = self._get_issues_to_compute(issue_types)
        self.issue_types.extend(to_compute_issues)
        print(
            f"Checking for {', '.join([issue_type.value for issue_type in to_compute_issues])} images ..."
        )

        # create issue managers
        self._set_issue_managers(to_compute_issues)

        for issue_type, issue_manager in self.issue_managers.items():
            issue_manager.find_issues(self.filepaths, self.info)

            # update issues, issue_summary and info
            self.issues = self.issues.join(issue_manager.issues, how="left")
            self.issue_summary = pd.concat(
                [self.issue_summary, issue_manager.summary], axis=0, ignore_index=True
            )
            self.info = {**self.info, **issue_manager.info}
        self.issue_summary = self.issue_summary.sort_values(
            by=["num_images"], ascending=False
        )

        return

    def _set_issue_managers(self, issue_types):
        image_property_issues = []
        for issue_type in self.issue_types:
            if issue_type.property:
                image_property_issues.append(issue_type)
            else:
                self.issue_managers[issue_type] = _IssueManagerFactory.from_str(
                    issue_type.value
                )

        if len(image_property_issues) > 0:
            if IMAGE_PROPERTY in self.issue_managers:
                self.issue_managers[IMAGE_PROPERTY].add_issues(image_property_issues)
            else:
                self.issue_managers[IMAGE_PROPERTY] = _IssueManagerFactory.from_str(
                    IMAGE_PROPERTY
                )(image_property_issues)

    def _get_topk_issues(self, topk, max_prevalence):
        topk_issues = []
        for idx, row in self.issue_summary.iterrows():
            if row["num_images"] / self.num_images * 100 < max_prevalence:
                topk_issues.append(row["issue_type"])
        return topk_issues[:topk]

    def report(self, topk=5, max_prevalence=50, verbose=False):
        topk_issues = self._get_topk_issues(topk, max_prevalence)
        topk_issue_summary = self.issue_summary[
            self.issue_summary["issue_type"].isin(topk_issues)
        ]
        if verbose:
            print("Issues in the dataset sorted by prevalence")
            print(self.issue_summary.to_markdown())
        else:
            print(f"Top issues in the dataset\n")
            print(topk_issue_summary.to_markdown(), "\n")
        topk_issues = self.issue_summary["issue_type"].tolist()[:topk]
        self.visualize(topk_issues)

    def _visualize(self, issue_type, num_images_per_issue, figsize):
        if issue_type in [IssueType.DARK_IMAGES.value, IssueType.LIGHT_IMAGES.value]:
            sorted_df = self.issues.sort_values(by=[f"{issue_type}_score"])
            sorted_df = sorted_df[sorted_df[f"{issue_type}_bool"] == 1]
            sorted_filepaths = sorted_df.index[:num_images_per_issue].tolist()
            VizManager.property_based(
                filepaths=sorted_filepaths,
                nrows=math.ceil(
                    min(num_images_per_issue, len(sorted_filepaths))
                    / self.config["viz_num_images_per_row"]
                ),
                ncols=self.config["viz_num_images_per_row"],
                figsize=figsize,
            )

    def visualize(self, issue_types, num_images_per_issue=4, figsize=(8, 8)):
        for issue_type in issue_types:
            print(f"\nTop {num_images_per_issue} images with {issue_type} issue")
            self._visualize(issue_type, num_images_per_issue, figsize)
