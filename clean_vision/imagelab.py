import math

import pandas as pd

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
        self.issues = pd.DataFrame(self.filepaths, columns=["image_path"])
        self.issue_types = list(IssueType)
        self.issue_managers = []
        # can be loaded from a file later
        self.config = {"viz_num_images_per_row": 4}

    def find_issues(self, issue_types=None):
        if issue_types is not None and len(issue_types) > 0:
            self.issue_types = []
            for issue_type_str, threshold in issue_types.items():
                issue_type = IssueType(issue_type_str)
                issue_type.threshold = threshold
                self.issue_types.append(issue_type)

        print(
            f"Checking for {', '.join([issue_type.value for issue_type in self.issue_types])} images ..."
        )

        # create issue managers
        self._set_issue_managers()

        for issue_manager in self.issue_managers:
            issue_manager.find_issues(self.filepaths, self.info)

            # update issues, issue_summary and info
            self.issues = self.issues.merge(
                issue_manager.issues, how="left", on="image_path"
            )
            self.issue_summary = pd.concat([self.issue_summary, issue_manager.summary])
            self.info = {**self.info, **issue_manager.info}
        self.issue_summary = self.issue_summary.sort_values(
            by=["num_images"], ascending=False
        )

        return

    def _set_issue_managers(self):
        image_property_issues = []
        for issue_type in self.issue_types:
            if issue_type.property:
                image_property_issues.append(issue_type)
            else:
                self.issue_managers.append(
                    _IssueManagerFactory.from_str(issue_type.value)
                )
        if len(image_property_issues) > 0:
            self.issue_managers.append(
                _IssueManagerFactory.from_str("ImageProperty")(image_property_issues)
            )

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

    def _visualize(self, issue_type, num_images_per_issue):
        if issue_type in [IssueType.DARK_IMAGES.value, IssueType.LIGHT_IMAGES.value]:
            sorted_df = self.issues.sort_values(by=[f"{issue_type}_score"])
            sorted_filepaths = (
                sorted_df["image_path"].head(num_images_per_issue).tolist()
            )
            VizManager.property_based(
                sorted_filepaths,
                math.ceil(num_images_per_issue / self.config["viz_num_images_per_row"]),
                self.config["viz_num_images_per_row"],
            )

    def visualize(self, issue_types, num_images_per_issue=4):
        for issue_type in issue_types:
            print(f"\nTop {num_images_per_issue} images with {issue_type} issue")
            self._visualize(issue_type, num_images_per_issue)
