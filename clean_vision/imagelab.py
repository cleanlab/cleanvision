import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from clean_vision.issue_types import IssueType
from clean_vision.utils.issue_manager_factory import _IssueManagerFactory
from clean_vision.utils.utils import get_filepaths


class Imagelab:
    def __init__(self, data_path):
        self.filepaths = get_filepaths(data_path)
        self.info = {}
        self.issue_summary = pd.DataFrame()
        self.issues = pd.DataFrame(self.filepaths, columns=["image_path"])
        self.issue_types = list(IssueType)
        self.issue_managers = []

    def find_issues(self, issue_types=None):
        if issue_types is not None and len(issue_types) > 0:
            self.issue_types = []
            for issue_type_str, threshold in issue_types.items():
                issue_type = IssueType(issue_type_str)
                issue_type.threshold = threshold
                self.issue_types.append(issue_type)

        print(
            f"Checking for {', '.join([issue_type.value for issue_type in self.issue_types])} images"
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

    def report(self, verbose=False):
        topk = 5

        if verbose:
            print("Issues in the dataset sorted by prevalence")
            print(self.issue_summary.to_markdown())
        else:

            print(f"Top {topk} issues in the dataset\n")
            print(self.issue_summary.head(topk).to_markdown())
        self._visualize()

    def _visualize(self, topk=5):

        topk_issues = self.issue_summary["issue_type"].tolist()[:topk]
        image_paths = []
        num_images_per_issue = 4
        for issue_type in topk_issues:
            sorted_df = self.issues.sort_values(by=[f"{issue_type}_score"])
            sorted_filepaths = (
                sorted_df["image_path"].head(num_images_per_issue).tolist()
            )
            image_paths.extend(sorted_filepaths)

        nrows, ncols = (
            int(len(image_paths) / num_images_per_issue),
            num_images_per_issue,
        )

        fig, axes2d = plt.subplots(
            nrows=max(nrows, 2), ncols=ncols, sharex=True, sharey=True, figsize=(10, 10)
        )

        img_idx = 0
        for i, row in enumerate(axes2d):
            if i == nrows:
                break
            for j, ax in enumerate(row):
                ax.imshow(Image.open(image_paths[img_idx]))
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticklabels([])
                ax.set_yticks([])
                ax.set_title(image_paths[img_idx].split("/")[-1], fontsize=5)
                if j == 0:
                    ax.set_ylabel(f"{topk_issues[i]}", rotation=0, labelpad=20)
                img_idx += 1

        plt.show()
