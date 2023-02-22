import multiprocessing
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
from cleanvision.utils.utils import get_max_n_jobs, get_is_issue_colname
from cleanvision.utils.constants import MAX_PROCS


def compute_scores(path: str, to_compute: List[str], properties) -> Dict[str, Any]:
    # compute_functions = {
    #     IssueType.DARK.value: calc_percentile_brightness,
    #     IssueType.LIGHT.value: calc_percentile_brightness,
    #     IssueType.ODD_ASPECT_RATIO.value: calc_aspect_ratio,
    #     IssueType.LOW_INFORMATION.value: calc_entropy,
    #     IssueType.BLURRY.value: calc_blurriness,
    #     IssueType.GRAYSCALE.value: calc_color_space,
    # }
    image = Image.open(path)
    results: Dict[str, Any] = {}
    results["path"] = path
    for issue_type in to_compute:
        results[issue_type] = properties[issue_type].calculate(image)
    return results


def compute_scores_wrapper(arg: Dict[str, Any]) -> Dict[str, Any]:
    to_compute = arg["to_compute"]
    path = arg["path"]
    properties = arg["image_properties"]
    return compute_scores(path, to_compute, properties)


# Combined all issues which are to be detected using image properties under one class to save time on loading image
@register_issue_manager(IMAGE_PROPERTY)
class ImagePropertyIssueManager(IssueManager):
    issue_name: str = IMAGE_PROPERTY
    visualization: str = "individual_images"

    def __init__(self) -> None:
        super().__init__()
        self.issue_types: List[str] = []
        self.params = self.get_default_params()
        self.image_properties = self._get_image_properties()

    def get_default_params(self) -> Dict[str, Any]:
        return {
            IssueType.DARK.value: {"threshold": 0.1},
            IssueType.LIGHT.value: {"threshold": 0.04},
            IssueType.ODD_ASPECT_RATIO.value: {"threshold": 0.4},
            IssueType.LOW_INFORMATION.value: {
                "threshold": 0.3,
                "normalizing_factor": 0.1,
            },
            IssueType.BLURRY.value: {"threshold": 0.06, "normalizing_factor": 0.001},
            IssueType.GRAYSCALE.value: {},
        }

    def update_params(self, params: Dict[str, Any]) -> None:
        for issue_type in self.params:
            non_none_params = {
                k: v for k, v in params.get(issue_type, {}).items() if v is not None
            }
            self.params[issue_type] = {**self.params[issue_type], **non_none_params}

    def _get_image_properties(self) -> Dict[str, Any]:
        return {
            IssueType.DARK.value: BrightnessProperty(IssueType.DARK.value),
            IssueType.LIGHT.value: BrightnessProperty(IssueType.LIGHT.value),
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
            score_column = self.image_properties[issue_type].score_column
            if score_column in imagelab_info[
                "statistics"
            ] or score_column in imagelab_info.get(issue_type, {}):
                defer_set.add(issue_type)

        # Add issues using same property
        if {IssueType.LIGHT.value, IssueType.DARK.value}.issubset(set(issue_types)):
            defer_set.add(IssueType.LIGHT.value)
        return defer_set

    def find_issues(
        self,
        *,
        params: Optional[Dict[str, Any]] = None,
        filepaths: Optional[List[str]] = None,
        imagelab_info: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().find_issues(**kwargs)
        assert params is not None
        assert imagelab_info is not None
        assert filepaths is not None

        self.issue_types = list(params.keys())
        self.update_params(params)

        defer_set = self._get_defer_set(self.issue_types, imagelab_info)

        to_be_computed = list(set(self.issue_types).difference(defer_set))

        agg_computations = {}
        if to_be_computed:
            if n_jobs is None:
                n_jobs = get_max_n_jobs()

            results: List[Any] = []
            if n_jobs == 1:
                for path in tqdm(filepaths):
                    results.append(
                        compute_scores(path, to_be_computed, self.image_properties)
                    )
            else:
                args = [
                    {
                        "to_compute": to_be_computed,
                        "path": path,
                        "image_properties": self.image_properties,
                    }
                    for i, path in enumerate(filepaths)
                ]
                chunksize = max(1, len(args) // MAX_PROCS)
                with multiprocessing.Pool(n_jobs) as p:
                    results = list(
                        tqdm(
                            p.imap_unordered(
                                compute_scores_wrapper, args, chunksize=chunksize
                            ),
                            total=len(filepaths),
                        )
                    )

                results = sorted(results, key=lambda r: r["path"])  # type:ignore

            agg_computations = self.aggregate_comp(results, to_be_computed)

        # update info
        self.update_info(agg_computations, imagelab_info)

        scores = self.compute_scores()

        self.update_issues(scores, filepaths)
        self.update_summary()
        return

    def update_issues(self, scores, filepaths):
        self.issues = pd.DataFrame(index=filepaths)
        for issue_type in self.issue_types:
            self.issues = self.issues.join(
                scores[issue_type].rename(f"{issue_type}_score")
            )
            is_issue = self.image_properties[issue_type].mark_issue(
                scores[issue_type], self.params[issue_type].get("threshold")
            )
            self.issues = self.issues.join(
                is_issue.rename(get_is_issue_colname(issue_type))
            )

    def compute_scores(self):
        scores = {}
        for issue_type in self.issue_types:
            score_column = self.image_properties[issue_type].score_column
            if score_column in self.info["statistics"]:
                values = self.info["statistics"][score_column]
            else:
                values = self.info[issue_type][score_column]

            scores[issue_type] = self.image_properties[issue_type].get_scores(
                raw_scores=values, **self.params[issue_type]
            )
        return scores

    def aggregate_comp(self, results: Dict[str, Any], issue_types):
        agg_computations: Dict[str, Any] = {
            issue_type: [] for issue_type in issue_types
        }
        paths = []
        for result in results:
            paths.append(result["path"])
            for issue_type in issue_types:
                agg_computations[issue_type].append(result[issue_type])
        for issue_type in issue_types:
            agg_computations[issue_type] = pd.DataFrame(
                agg_computations[issue_type], index=paths
            )
        return agg_computations

    def update_info(self, agg_computations: Dict[str, Any], imagelab_info) -> None:
        for issue_type in self.issue_types:
            property_name = self.image_properties[issue_type].name
            if issue_type in agg_computations:
                comp = agg_computations[issue_type]
                self.info[issue_type] = {}

                for column_name in comp.columns:
                    if column_name == property_name:
                        self.info["statistics"][property_name] = comp[property_name]
                    else:
                        self.info[issue_type][column_name] = comp[column_name]
            else:
                if property_name not in self.info["statistics"]:
                    self.info["statistics"][property_name] = imagelab_info[
                        "statistics"
                    ][property_name]
                self.info[issue_type] = imagelab_info.get(issue_type, {})

        # todo: revisit when there are more issues using same properties like light and dark
        if (
            IssueType.LIGHT.value in self.issue_types
            and not self.info[IssueType.LIGHT.value]
        ):
            self.info[IssueType.LIGHT.value] = self.info[IssueType.DARK.value]
        if (
            IssueType.DARK.value in self.issue_types
            and not self.info[IssueType.DARK.value]
        ):
            self.info[IssueType.DARK.value] = self.info[IssueType.LIGHT.value]

    def update_summary(self) -> None:
        summary_dict = {}
        for issue_type in self.issue_types:
            summary_dict[issue_type] = self._compute_summary(
                self.issues[get_is_issue_colname(issue_type)]
            )
        summary_df = pd.DataFrame.from_dict(summary_dict, orient="index")
        self.summary = summary_df.reset_index()
        self.summary = self.summary.rename(columns={"index": "issue_type"})
        self.summary = self.summary.astype({"num_images": int, "issue_type": str})
