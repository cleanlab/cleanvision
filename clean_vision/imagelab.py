import math

import pandas as pd

from clean_vision.issue_managers import IssueManagerFactory, IssueType
from clean_vision.utils.constants import IMAGE_PROPERTY, IMAGE_PROPERTY_ISSUE_TYPES_LIST
from clean_vision.utils.utils import get_filepaths
from clean_vision.utils.viz_manager import VizManager


class Imagelab:
    def __init__(self, data_path):
        self.filepaths = get_filepaths(data_path)
        self.num_images = len(self.filepaths)
        self.info = {}  # todo initialize with stats
        self.issue_summary = pd.DataFrame()
        # self.issues = pd.DataFrame(self.filepaths, columns=["image_path"])
        self.issues = pd.DataFrame(index=self.filepaths)
        self.issue_types = []
        self.issue_managers = {}
        # can be loaded from a file later
        self._set_default_config()

    def list_default_issue_types(self):
        print(*self.config["default_issue_types"], sep="\n")

    def list_possible_issue_types(self):
        print(*list(IssueType), sep="\n")

    def _set_default_config(self):
        self.config = {
            "visualize_num_images_per_row": 4,
            "report_num_top_issues_values": [3, 5, 10, len(self.issue_types)],
            "report_examples_per_issue_values": [4, 8, 16, 32],
            "report_max_prevalence": 0.5,
            "default_issue_types": [IssueType.DARK, IssueType.LIGHT],
        }

    def _get_issues_to_compute(self, issue_types):
        if issue_types is None or len(issue_types) == 0:
            all_issues = self.config["default_issue_types"]
        else:
            all_issues = []
            for issue_type_str, hyperparameters in issue_types.items():
                issue_type = IssueType(issue_type_str)
                issue_type.set_hyperparameters(hyperparameters)
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
        self._set_issue_managers(
            to_compute_issues,
        )

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
        image_property_issues_types = []
        for issue_type in issue_types:
            if issue_type.value in IMAGE_PROPERTY_ISSUE_TYPES_LIST:
                image_property_issues_types.append(issue_type)
            else:
                self.issue_managers[issue_type] = IssueManagerFactory.from_str(
                    issue_type.value
                )()

        if len(image_property_issues_types) > 0:
            if IMAGE_PROPERTY in self.issue_managers:
                # todo: do not re-use the same object, create a new issue_manager
                self.issue_managers[IMAGE_PROPERTY].add_issue_types(
                    image_property_issues_types
                )
            else:
                self.issue_managers[IMAGE_PROPERTY] = IssueManagerFactory.from_str(
                    IMAGE_PROPERTY
                )(image_property_issues_types)

    def _get_topk_issues(self, num_top_issues, max_prevalence):
        topk_issues = []
        for idx, row in self.issue_summary.iterrows():
            if row["num_images"] / self.num_images < max_prevalence:
                topk_issues.append(row["issue_type"])
            else:
                print(
                    f"Dropping {row['issue_type']} as a possible issue since it's present in more than {max_prevalence * 100} percent of the images"
                )
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

        print(f"Top issues in the dataset\n")
        print(top_issue_summary.to_markdown(), "\n")

        self.visualize(top_issues, report_args["examples_per_issue"])

    def _visualize(self, issue_type_str, examples_per_issue, figsize):
        if issue_type_str in [
            IssueType.DARK_IMAGES.value,
            IssueType.LIGHT_IMAGES.value,
            IssueType.CUSTOM_IMAGES.value,
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


# registration method for issues
# print statement for max_prevalence
