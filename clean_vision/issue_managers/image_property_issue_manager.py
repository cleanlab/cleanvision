import pandas as pd
from PIL import Image
from tqdm import tqdm

from clean_vision.issue_managers.base import IssueManager
from clean_vision.issue_managers.image_property_helpers import BrightnessHelper
from clean_vision.issue_types import IssueType


class ImagePropertyIssueManager(IssueManager):
    def __init__(self, issue_types):
        super().__init__()
        self.issue_types = issue_types
        self.issue_helpers = {
            IssueType.DARK_IMAGES: BrightnessHelper(IssueType.DARK_IMAGES),
            IssueType.LIGHT_IMAGES: BrightnessHelper(IssueType.LIGHT_IMAGES),
        }

    def _get_skip_set(self):
        skip_set = set()
        if set([IssueType.LIGHT_IMAGES, IssueType.DARK_IMAGES]).issubset(
            set(self.issue_types)
        ):
            skip_set.add(IssueType.LIGHT_IMAGES)
        return skip_set

    def find_issues(self, filepaths, imagelab_info):
        self.issues = pd.DataFrame(filepaths, columns=["image_path"])

        raw_scores = {issue_type.property: [] for issue_type in self.issue_types}
        skip_set = self._get_skip_set()
        for path in tqdm(filepaths):
            image = Image.open(path)
            for issue_type in self.issue_types:
                if issue_type not in skip_set:
                    raw_scores[issue_type.property].append(
                        self.issue_helpers[issue_type].calculate(image)
                    )

        for issue_type in self.issue_types:
            if issue_type.property not in self.info:
                self.info[issue_type.property] = raw_scores[issue_type.property]

            scores = self.issue_helpers[issue_type].normalize(
                raw_scores[issue_type.property]
            )
            self.issues[f"{issue_type}_score"] = scores
            self.issues[f"{issue_type}_bool"] = self.issue_helpers[
                issue_type
            ].mark_issue(scores, issue_type.threshold)

            summary = self._compute_summary(self.issues[f"{issue_type}_bool"])
            summary = pd.DataFrame(
                [[issue_type.value, summary["num_images"]]],
                columns=self.summary.columns,
            )
            self.summary = pd.concat([self.summary, summary], ignore_index=True)

        return
