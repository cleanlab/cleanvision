from typing import Dict, Any, List, Set, Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm

from cleanvision.issue_managers import register_issue_manager, IssueType
from cleanvision.issue_managers.image_property import (
    BrightnessProperty,
    AspectRatioProperty,
    EntropyProperty,
    BlurrinessProperty,
    ColorSpaceProperty,
)
from cleanvision.utils.base_issue_manager import IssueManager
from cleanvision.utils.constants import IMAGE_PROPERTY

import multiprocessing
import psutil
from cleanvision.issue_managers.image_property import (
    calc_brightness,
    calc_aspect_ratio,
    calc_entropy,
    calc_blurriness,
    calc_color_space,
)


def compute_scores(arg):
    compute_functions = {
        IssueType.DARK.value: calc_brightness,
        IssueType.LIGHT.value: calc_brightness,
        IssueType.ODD_ASPECT_RATIO.value: calc_aspect_ratio,
        IssueType.LOW_INFORMATION.value: calc_entropy,
        IssueType.BLURRY.value: calc_blurriness,
        IssueType.GRAYSCALE.value: calc_color_space,
    }
    to_compute = arg["to_compute"]
    path = arg["path"]
    image = Image.open(path)
    results = {}
    results["path"] = path
    for issue_type in to_compute:
        results[issue_type] = compute_functions[issue_type](image)
    return results



# Combined all issues which are to be detected using image properties under one class to save time on loading image
@register_issue_manager(IMAGE_PROPERTY)
class ImagePropertyIssueManager(IssueManager):
    issue_name: str = IMAGE_PROPERTY
    visualization: str = "individual_images"

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.issue_types: List[str] = list(params.keys())
        self.set_params(params)
        self.image_properties = self._get_image_properties()

    def get_default_params(self) -> Dict[str, Any]:
        return {
            IssueType.DARK.value: {"threshold": 0.22},
            IssueType.LIGHT.value: {"threshold": 0.05},
            IssueType.ODD_ASPECT_RATIO.value: {"threshold": 0.5},
            # todo: check low complexity params on a different dataset
            IssueType.LOW_INFORMATION.value: {
                "threshold": 0.3,
                "normalizing_factor": 0.1,
            },
            IssueType.BLURRY.value: {"threshold": 0.3, "normalizing_factor": 0.001},
            IssueType.GRAYSCALE.value: {},
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        self.params = self.get_default_params()
        for issue_type in self.params:
            non_none_params = {
                k: v for k, v in params.get(issue_type, {}).items() if v is not None
            }
            self.params[issue_type] = {**self.params[issue_type], **non_none_params}

    def _get_image_properties(self) -> Dict[str, Any]:
        return {
            IssueType.DARK.value: BrightnessProperty(IssueType.DARK),
            IssueType.LIGHT.value: BrightnessProperty(IssueType.LIGHT),
            IssueType.ODD_ASPECT_RATIO.value: AspectRatioProperty(),
            IssueType.LOW_INFORMATION.value: EntropyProperty(),
            IssueType.BLURRY.value: BlurrinessProperty(),
            IssueType.GRAYSCALE.value: ColorSpaceProperty(),
        }

    def _get_defer_set(
        self, issue_types: List[str], imagelab_info: Dict[str, Any]
    ) -> Set[str]:
        defer_set = set()

        # Add precomputed issues to defer set
        for issue_type in issue_types:
            image_property = self.image_properties[issue_type].name
            if image_property in imagelab_info[
                "statistics"
            ] or image_property in imagelab_info.get(
                issue_type, {}
            ):  # todo: check not needed as properties always added in imagelab["statistics"]
                defer_set.add(issue_type)

        # Add issues using same property
        if {IssueType.LIGHT.value, IssueType.DARK.value}.issubset(set(issue_types)):
            defer_set.add(IssueType.LIGHT.value)
        return defer_set

    def find_issues_multi(
        self,
        *,
        filepaths: Optional[List[str]] = None,
        imagelab_info: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().find_issues(**kwargs)
        assert imagelab_info is not None
        assert filepaths is not None

        defer_set = self._get_defer_set(self.issue_types, imagelab_info)

        to_be_computed = list(set(self.issue_types).difference(defer_set))
        raw_scores: Dict[str, Any] = {issue_type: [] for issue_type in to_be_computed}
        if to_be_computed:
            args = [
                {"to_compute": to_be_computed, "path": path}
                for i, path in enumerate(filepaths)
            ]
            if n_jobs is None:
                n_jobs = psutil.cpu_count(logical=False)
            with multiprocessing.Pool(n_jobs) as p:
                computed_results = list(
                    tqdm(
                        p.imap_unordered(compute_scores, args, chunksize=10),
                        total=len(filepaths),
                    )
                )

            computed_results = sorted(computed_results, key=lambda r: r["path"])
            for result in computed_results:
                for issue_type in to_be_computed:
                    raw_scores[issue_type].append(result[issue_type])

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
            self.issues[f"{issue_type}_score"] = scores
            self.issues[f"{issue_type}_bool"] = self.image_properties[
                issue_type
            ].mark_issue(scores, self.params[issue_type].get("threshold"))

            summary_dict[issue_type] = self._compute_summary(
                self.issues[f"{issue_type}_bool"]
            )

        # update issues and summary
        self.update_summary(summary_dict)
        return

    def find_issues(
        self,
        *,
        filepaths: Optional[List[str]] = None,
        imagelab_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().find_issues(**kwargs)
        assert imagelab_info is not None
        assert filepaths is not None

        defer_set = self._get_defer_set(self.issue_types, imagelab_info)

        to_be_computed = list(set(self.issue_types).difference(defer_set))
        raw_scores: Dict[str, Any] = {issue_type: [] for issue_type in to_be_computed}
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
                raw_scores=property_values, **self.params[issue_type]
            )

            # Update issues
            self.issues[f"{issue_type}_score"] = scores
            self.issues[f"{issue_type}_bool"] = self.image_properties[
                issue_type
            ].mark_issue(scores, self.params[issue_type].get("threshold"))

            summary_dict[issue_type] = self._compute_summary(
                self.issues[f"{issue_type}_bool"]
            )

        # update issues and summary
        self.update_summary(summary_dict)
        return

    def update_info(self, raw_scores: Dict[str, Any]) -> None:
        for issue_type, scores in raw_scores.items():
            # todo: add a way to update info for image properties which are not stats
            if self.image_properties[issue_type].name is not None:
                self.info["statistics"][self.image_properties[issue_type].name] = scores

    def update_summary(self, summary_dict: Dict[str, Any]) -> None:
        summary_df = pd.DataFrame.from_dict(summary_dict, orient="index")
        self.summary = summary_df.reset_index()
        self.summary = self.summary.rename(columns={"index": "issue_type"})
        self.summary = self.summary.astype({"num_images": int, "issue_type": str})
