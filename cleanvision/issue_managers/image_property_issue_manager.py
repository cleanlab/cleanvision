import pandas as pd
from PIL import Image
from tqdm import tqdm

from cleanvision.issue_managers import register_issue_manager, IssueType
from cleanvision.issue_managers.image_property_helpers import BrightnessHelper
from cleanvision.utils.base_issue_manager import IssueManager
from cleanvision.utils.constants import IMAGE_PROPERTY


# Combined all issues which are to be detected using image properties under one class to save time on loading image
@register_issue_manager(IMAGE_PROPERTY)
class ImagePropertyIssueManager(IssueManager):
    issue_name = IMAGE_PROPERTY
    visualization = "property_based"

    def __init__(self, params):
        self.thresholds = {}
        super().__init__(params)
        self.issue_helpers = self._get_default_issue_helpers()

    def _get_default_thresholds(self):
        return {IssueType.DARK: 0.22, IssueType.LIGHT: 0.05}

    def set_params(self, image_property_params):
        # set issue_types
        self.issue_types = list(image_property_params.keys())

        # set thresholds
        default_thresholds = self._get_default_thresholds()

        for issue_type, issue_params in image_property_params.items():
            self.thresholds[issue_type] = issue_params.get("threshold", default_thresholds[issue_type])

    def _get_default_issue_helpers(self):
        return {
            IssueType.DARK: BrightnessHelper(IssueType.DARK),
            IssueType.LIGHT: BrightnessHelper(IssueType.LIGHT),
        }

    def _get_defer_set(self, imagelab_info):
        defer_set = set()

        # Add precomputed issues to defer set
        for issue_type in self.issue_types:
            image_property = self.issue_helpers[issue_type].image_property
            if image_property in imagelab_info[
                "statistics"
            ] or image_property in imagelab_info.get(issue_type.value, {}):
                defer_set.add(issue_type)

        # Add issues using same property
        if set([IssueType.LIGHT, IssueType.DARK]).issubset(set(self.issue_types)):
            defer_set.add(IssueType.LIGHT)
        return defer_set

    def _get_to_be_computed_issue_types(self):
        to_be_computed = [
            issue_type
            for issue_type, computed in self.issue_types_computed.items()
            if not computed
        ]
        return to_be_computed

    def find_issues(self, filepaths, imagelab_info):
        defer_set = self._get_defer_set(imagelab_info)

        to_be_computed = list(set(self.issue_types).difference(defer_set))
        raw_scores = {issue_type: [] for issue_type in to_be_computed}
        if to_be_computed:
            for path in tqdm(filepaths):
                image = Image.open(path)
                for issue_type in to_be_computed:
                    raw_scores[issue_type].append(
                        self.issue_helpers[issue_type].calculate(image)
                    )

        # update info
        self.update_info(raw_scores)

        # Init issues, summary
        self.issues = pd.DataFrame(index=filepaths)
        summary_dict = {}

        for issue_type in self.issue_types:
            image_property = self.issue_helpers[issue_type].image_property
            if image_property in imagelab_info["statistics"]:
                property_values = imagelab_info["statistics"][image_property]
            else:
                property_values = self.info["statistics"][image_property]

            scores = self.issue_helpers[issue_type].normalize(property_values)

            # Update issues
            self.issues[f"{issue_type.value}_score"] = scores
            self.issues[f"{issue_type.value}_bool"] = self.issue_helpers[
                issue_type
            ].mark_issue(scores, self.thresholds[issue_type])

            summary_dict[issue_type.value] = self._compute_summary(
                self.issues[f"{issue_type.value}_bool"]
            )

        # update issues and summary
        self.update_summary(summary_dict)
        return

    def update_info(self, raw_scores):
        for issue_type, scores in raw_scores.items():
            # todo: add a way to update info for image properties which are not stats
            if self.issue_helpers[issue_type].image_property is not None:
                self.info["statistics"][
                    self.issue_helpers[issue_type].image_property
                ] = scores

    def update_summary(self, summary_dict: dict):
        summary_df = pd.DataFrame.from_dict(summary_dict, orient="index")
        self.summary = summary_df.reset_index(names="issue_type")
