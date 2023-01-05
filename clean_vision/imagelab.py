import math

import pandas as pd

from clean_vision.issue_managers import IssueType, IssueManagerFactory
from clean_vision.utils.constants import IMAGE_PROPERTY, IMAGE_PROPERTY_ISSUE_TYPES_LIST
from clean_vision.utils.utils import get_filepaths
from clean_vision.utils.viz_manager import VizManager


class Imagelab:
    def __init__(self, data_path):
        self.filepaths = get_filepaths(data_path)
        self.num_images = len(self.filepaths)
        self.info = {}  # todo initialize with stats
        self.issue_summary = pd.DataFrame()
        self.issues = pd.DataFrame(index=self.filepaths)
        self.issue_types = []
        self.issue_managers = {}
        # can be loaded from a file later
        self.config = self._set_default_config()

    def _set_default_config(self):
        return {
            "visualize_num_images_per_row": 4,
            "report_num_top_issues_values": [3, 5, 10, len(self.issue_types)],
            "report_examples_per_issue_values": [4, 8, 16, 32],
            "report_max_prevalence": 0.5,
            "default_issue_types": [IssueType.DARK, IssueType.LIGHT],
        }

    def list_default_issue_types(self):
        print(*self.config["default_issue_types"], sep="\n")

    def list_possible_issue_types(self):
        print(*list(IssueType), sep="\n")

    def _get_issues_to_compute(self, issue_types_with_params):
        if issue_types_with_params is None or len(issue_types_with_params) == 0:
            to_compute_issues_with_params = {
                issue_type: {} for issue_type in self.config["default_issue_types"]
            }
        else:
            to_compute_issues_with_params = issue_types_with_params
        return to_compute_issues_with_params

    def find_issues(self, issue_types=None):
        to_compute_issues_with_params = self._get_issues_to_compute(issue_types)
        print(
            f"Checking for {', '.join([issue_type for issue_type in to_compute_issues_with_params.keys()])} images ..."
        )

        # update issue_types
        self.issue_types = list(
            set(self.issue_types).union(set(to_compute_issues_with_params.keys()))
        )

        # create issue managers
        self._set_issue_managers(
            to_compute_issues_with_params,
        )

        # find issues
        for issue_manager in self.issue_managers.values():
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

    def _get_image_property_issues(self, issue_types_with_params):
        image_property_issues = {}
        for issue_type, params in issue_types_with_params.items():
            if issue_type.value in IMAGE_PROPERTY_ISSUE_TYPES_LIST:
                image_property_issues[issue_type] = params
        return image_property_issues

    def _set_issue_managers(self, issue_types_with_params):
        image_property_issue_types = self._get_image_property_issues(
            issue_types_with_params
        )
        self.issue_managers[IMAGE_PROPERTY] = IssueManagerFactory.from_str(
            IMAGE_PROPERTY
        )(image_property_issue_types)
        for issue_type, params in issue_types_with_params:
            if issue_type not in image_property_issue_types:
                self.issue_managers[issue_type] = IssueManagerFactory.from_str(
                    issue_type.value
                )(params)

    def _get_topk_issues(self, num_top_issues, max_prevalence):
        topk_issues = []
        for row in self.issue_summary.itertuples(index=False):
            if getattr(row, "num_images") / self.num_images < max_prevalence:
                topk_issues.append(getattr(row, "issue_type"))
        return topk_issues[:num_top_issues]

    def _get_report_args(self, verbosity, user_supplied_args):
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
        num_top_issues=None,
        max_prevalence=None,
        examples_per_issue=None,
        verbosity=1,
    ):
        assert isinstance(verbosity, int) and 0 < verbosity < 5
        user_supplied_args = locals()
        report_args = self._get_report_args(verbosity, user_supplied_args)

        top_issues = self._get_topk_issues(
            report_args["num_top_issues"], report_args["max_prevalence"]
        )
        top_issue_summary = self.issue_summary[
            self.issue_summary["issue_type"].isin(top_issues)
        ]

        print("Top issues in the dataset\n")
        print(top_issue_summary.to_markdown(), "\n")

        self.visualize(top_issues, report_args["examples_per_issue"])

    def _visualize(self, issue_type_str, examples_per_issue, figsize):
        if issue_type_str in [
            IssueType.DARK_IMAGES.value,
            IssueType.LIGHT_IMAGES.value,
        ]:
            sorted_df = self.issues.sort_values(by=[f"{issue_type_str}_score"])
            sorted_df = sorted_df[sorted_df[f"{issue_type_str}_bool"] == 1]
            if len(sorted_df) < examples_per_issue:
                print(
                    f"Found only {len(sorted_df)} examples of {issue_type_str} issue in the dataset."
                )
            else:
                print(f"\nTop {examples_per_issue} images with {issue_type_str} issue")
            sorted_filepaths = sorted_df.index[:examples_per_issue].tolist()
            VizManager.property_based(
                filepaths=sorted_filepaths,
                nrows=math.ceil(
                    min(examples_per_issue, len(sorted_filepaths))
                    / self.config["visualize_num_images_per_row"]
                ),
                ncols=self.config["visualize_num_images_per_row"],
                figsize=figsize,
            )

    def visualize(self, issue_types, examples_per_issue=4, figsize=(8, 8)):
        for issue_type in issue_types:
            self._visualize(issue_type, examples_per_issue, figsize)
