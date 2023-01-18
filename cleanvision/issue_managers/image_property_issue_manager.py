import pandas as pd
from PIL import Image
from tqdm import tqdm

from cleanvision.issue_managers import register_issue_manager, IssueType
from cleanvision.issue_managers.image_property import (
    BrightnessProperty,
    AspectRatioProperty,
    EntropyProperty,
    BlurrinessProperty,
)
from cleanvision.utils.base_issue_manager import IssueManager
from cleanvision.utils.constants import IMAGE_PROPERTY


# Combined all issues which are to be detected using image properties under one class to save time on loading image
@register_issue_manager(IMAGE_PROPERTY)
class ImagePropertyIssueManager(IssueManager):
    issue_name = IMAGE_PROPERTY
    visualization = "individual_images"

    def __init__(self, params):
        super().__init__(params)
        self.issue_types = list(self.params.keys())
        self.image_properties = self._get_image_properties()

    def get_default_params(self):
        return {
            IssueType.DARK: {"threshold": 0.22},
            IssueType.LIGHT: {"threshold": 0.05},
            IssueType.ODD_ASPECT_RATIO: {"threshold": 0.5},
            # todo: check low complexity params on a different dataset
            IssueType.LOW_INFORMATION: {"threshold": 0.3, "normalizing_factor": 0.1},
            IssueType.BLURRED: {"threshold": 0.3, "normalizing_factor": 0.001},
        }

    def set_params(self, params):
        update_params = {}
        for issue_type, issue_params in params.items():
            non_none_params = {k: v for k, v in issue_params.items() if v is not None}
            update_params[issue_type] = {**self.params[issue_type], **non_none_params}
        self.params = update_params

    def _get_image_properties(self):
        return {
            IssueType.DARK: BrightnessProperty(IssueType.DARK),
            IssueType.LIGHT: BrightnessProperty(IssueType.LIGHT),
            IssueType.ODD_ASPECT_RATIO: AspectRatioProperty(),
            IssueType.LOW_INFORMATION: EntropyProperty(),
            IssueType.BLURRED: BlurrinessProperty(),
        }

    def _get_defer_set(self, imagelab_info):
        defer_set = set()

        # Add precomputed issues to defer set
        for issue_type in self.issue_types:
            image_property = self.image_properties[issue_type].name
            if image_property in imagelab_info[
                "statistics"
            ] or image_property in imagelab_info.get(issue_type.value, {}):
                defer_set.add(issue_type)

        # Add issues using same property
        if {IssueType.LIGHT, IssueType.DARK}.issubset(set(self.issue_types)):
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
                        self.image_properties[issue_type].calculate(image)
                    )

        # update info
        self.update_info(raw_scores)

        # Init issues, summary
        self.issues = pd.DataFrame(index=filepaths)
        summary_dict = {}

        for issue_type in self.issue_types:
            image_property = self.image_properties[issue_type].name
            if image_property in imagelab_info["statistics"]:
                property_values = imagelab_info["statistics"][image_property]
            else:
                property_values = self.info["statistics"][image_property]

            scores = self.image_properties[issue_type].get_scores(
                property_values, **self.params[issue_type]
            )

            # Update issues
            self.issues[f"{issue_type.value}_score"] = scores
            self.issues[f"{issue_type.value}_bool"] = self.image_properties[
                issue_type
            ].mark_issue(scores, self.params[issue_type]["threshold"])

            summary_dict[issue_type.value] = self._compute_summary(
                self.issues[f"{issue_type.value}_bool"]
            )

        # update issues and summary
        self.update_summary(summary_dict)
        return

    def update_info(self, raw_scores):
        for issue_type, scores in raw_scores.items():
            # todo: add a way to update info for image properties which are not stats
            if self.image_properties[issue_type].name is not None:
                self.info["statistics"][self.image_properties[issue_type].name] = scores

    def update_summary(self, summary_dict: dict):
        summary_df = pd.DataFrame.from_dict(summary_dict, orient="index")
        self.summary = summary_df.reset_index(names="issue_type")
