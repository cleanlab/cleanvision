import pandas as pd
from PIL import Image
from tqdm import tqdm

from clean_vision.issue_managers import register_issue_manager, IssueType
from clean_vision.utils.base_issue_manager import IssueManager
from clean_vision.utils.constants import IMAGE_PROPERTY
from clean_vision.utils.image_property_helpers import BrightnessHelper


# Combined all issues which are to be detected using image properties under one class to save time on loading image
@register_issue_manager(IMAGE_PROPERTY)
class ImagePropertyIssueManager(IssueManager):
    def __init__(self, issue_types):
        super().__init__()
        self.issue_types_computed = {
            issue_type: False for issue_type in issue_types
        }  # Flag for computed issues, False if not computed
        self.issue_helpers = self._get_default_issue_helpers()

    @classmethod
    @property
    def issue_name(cls):
        return IMAGE_PROPERTY

    def _get_default_issue_helpers(self):
        return {
            IssueType.DARK: BrightnessHelper(IssueType.DARK),
            IssueType.LIGHT: BrightnessHelper(IssueType.LIGHT),
        }

    def _get_defer_set(self, to_be_computed):
        skip_set = set()
        for issue_type in to_be_computed:
            if issue_type.name == IssueType.LIGHT.name:
                # add light images to skip set if dark images already computed
                # or dark images is also present in to_be_computed set
                if (
                    self.issue_types_computed.get(IssueType.DARK, False)
                    or IssueType.DARK in to_be_computed
                ):
                    skip_set.add(issue_type)
            elif issue_type.name == IssueType.DARK.name:
                # add dark images to skip set if light images already computed
                if self.issue_types_computed.get(IssueType.LIGHT, False):
                    skip_set.add(issue_type)
        return skip_set

    def _get_to_be_computed_issue_types(self):
        to_be_computed = [
            issue_type
            for issue_type, computed in self.issue_types_computed.items()
            if not computed
        ]
        return to_be_computed

    def find_issues(self, filepaths, imagelab_info):

        to_be_computed = self._get_to_be_computed_issue_types()
        defer_set = self._get_defer_set(to_be_computed)

        # Calculate raw scores for each issue, re-use previously computed properties
        raw_scores = {issue_type.property: [] for issue_type in to_be_computed}

        if len(set(to_be_computed).difference(defer_set)) > 0:
            # todo test this bit
            for path in tqdm(filepaths):
                image = Image.open(path)
                for issue_type in to_be_computed:
                    if issue_type not in defer_set:
                        raw_scores[issue_type.property].append(
                            self.issue_helpers[issue_type].calculate(image)
                        )

        # Init issues, summary, info
        self.issues = pd.DataFrame(index=filepaths)
        summary_dict = {}
        self.info = {}

        for issue_type in to_be_computed:
            if issue_type.property in imagelab_info:
                # re-use imagelab info
                property_values = imagelab_info[issue_type.property]
            elif issue_type.property in self.info:
                # re-use issue_manager info
                property_values = self.info[issue_type.property]
            else:
                # update info
                property_values = raw_scores[issue_type.property]
                self.info[issue_type.property] = property_values

            scores = self.issue_helpers[issue_type].normalize(property_values)

            # Update issues
            self.issues[f"{issue_type}_score"] = scores
            self.issues[f"{issue_type}_bool"] = self.issue_helpers[
                issue_type
            ].mark_issue(scores, issue_type.threshold)

            summary_dict[issue_type.value] = self._compute_summary(
                self.issues[f"{issue_type}_bool"]
            )

        # update issues and summary
        self.update_summary(summary_dict)
        self._mark_computed(to_be_computed)
        return

    def update_summary(self, summary):
        summary_df = pd.DataFrame.from_dict(summary, orient="index")
        self.summary = summary_df.reset_index(names="issue_type")

    def _mark_computed(self, issue_types):
        for issue_type in issue_types:
            self.issue_types_computed[issue_type] = True

    def add_issue_types(self, issue_types):
        for issue_type in issue_types:
            if issue_type not in self.issue_types_computed:
                self.issue_types_computed[issue_type] = False
