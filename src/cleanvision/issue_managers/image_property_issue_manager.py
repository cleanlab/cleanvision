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
    ImageProperty,
)
from cleanvision.utils.base_issue_manager import IssueManager
from cleanvision.utils.constants import (
    IMAGE_PROPERTY,
    MAX_PROCS,
    IMAGE_PROPERTY_ISSUE_TYPES_LIST,
)
from cleanvision.utils.utils import get_max_n_jobs, get_is_issue_colname, update_df


def compute_scores(
    path: str, to_compute: List[str], properties: Dict[str, ImageProperty]
) -> Dict[str, Any]:
    image = Image.open(path)
    results: Dict[str, Any] = {}
    results["path"] = path
    for issue_type in to_compute:
        results = {**results, **properties[issue_type].calculate(image)}
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
            IssueType.DARK.value: {"threshold": 0.37},
            IssueType.LIGHT.value: {"threshold": 0.05},
            IssueType.ODD_ASPECT_RATIO.value: {"threshold": 0.35},
            IssueType.LOW_INFORMATION.value: {
                "threshold": 0.3,
                "normalizing_factor": 0.1,
            },
            IssueType.BLURRY.value: {"threshold": 0.15, "normalizing_factor": 0.01},
            IssueType.GRAYSCALE.value: {},
        }

    def update_params(self, params: Dict[str, Any]) -> None:
        for issue_type in self.params:
            non_none_params = {
                k: v for k, v in params.get(issue_type, {}).items() if v is not None
            }
            self.params[issue_type] = {**self.params[issue_type], **non_none_params}

    def _get_image_properties(self) -> Dict[str, ImageProperty]:
        return {
            IssueType.DARK.value: BrightnessProperty(IssueType.DARK.value),
            IssueType.LIGHT.value: BrightnessProperty(IssueType.LIGHT.value),
            IssueType.ODD_ASPECT_RATIO.value: AspectRatioProperty(),
            IssueType.LOW_INFORMATION.value: EntropyProperty(),
            IssueType.BLURRY.value: BlurrinessProperty(),
            IssueType.GRAYSCALE.value: ColorSpaceProperty(),
        }

    def _get_defer_set(
        self, issue_types: List[str], agg_computations: pd.DataFrame
    ) -> Set[str]:
        defer_set = set()

        # Add precomputed issues to defer set
        for issue_type in issue_types:
            score_columns = self.image_properties[issue_type].score_columns
            if set(score_columns).issubset(agg_computations.columns):
                defer_set.add(issue_type)

        # Add issues using same property
        if {IssueType.LIGHT.value, IssueType.DARK.value}.issubset(set(issue_types)):
            defer_set.add(IssueType.LIGHT.value)

        return defer_set

    def _get_additional_to_compute_set(self, issue_types):
        additional_set = []
        if (
            IssueType.BLURRY.value in issue_types
            and IssueType.DARK.value not in issue_types
        ):
            additional_set.append(IssueType.DARK.value)
        return additional_set

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
        self.issues = pd.DataFrame(index=filepaths)
        additional_set = self._get_additional_to_compute_set(self.issue_types)

        self.update_params(params)

        agg_computations = self._get_prev_computations(filepaths, imagelab_info)
        defer_set = self._get_defer_set(
            self.issue_types + additional_set, agg_computations
        )

        to_be_computed = list(
            set(self.issue_types + additional_set).difference(defer_set)
        )
        new_computations = pd.DataFrame(index=filepaths)
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

            new_computations = self.aggregate(results)

        agg_computations = update_df(agg_computations, new_computations)

        ordered_issue_types = self._get_dependency_sorted(self.issue_types)

        self.update_issues(agg_computations, ordered_issue_types)
        self.update_info(agg_computations)
        self.update_summary()
        return

    def _get_dependency_sorted(self, issue_types):
        # todo: remove this hacky way, add an ordering
        return sorted(issue_types, reverse=True)

    def update_issues(self, agg_computations: pd.DataFrame, issue_types) -> None:
        for issue_type in issue_types:
            score_column_names = self.image_properties[issue_type].score_columns

            # todo: this is hacky
            if issue_type == IssueType.BLURRY.value:
                all_info_df = agg_computations.join(self.issues)
                if not set(score_column_names).issubset(all_info_df.columns):
                    dark_issue_scores = self.image_properties[
                        IssueType.DARK.value
                    ].get_scores(
                        raw_scores=all_info_df[
                            self.image_properties[IssueType.DARK.value].score_columns
                        ],
                        **self.params[IssueType.DARK.value],
                        issue_type=IssueType.DARK.value,
                    )
                    is_dark_issue = self.image_properties[
                        IssueType.DARK.value
                    ].mark_issue(
                        dark_issue_scores,
                        self.params[IssueType.DARK.value].get("threshold"),
                        IssueType.DARK.value,
                    )
                    all_info_df = all_info_df.join(dark_issue_scores)
                    all_info_df = all_info_df.join(is_dark_issue)
                score_columns = all_info_df[score_column_names]

            else:
                score_columns = agg_computations[score_column_names]

            issue_scores = self.image_properties[issue_type].get_scores(
                raw_scores=score_columns,
                **self.params[issue_type],
                issue_type=issue_type,
            )
            is_issue = self.image_properties[issue_type].mark_issue(
                issue_scores, self.params[issue_type].get("threshold"), issue_type
            )
            self.issues = self.issues.join(issue_scores)
            self.issues = self.issues.join(is_issue)

    def _get_prev_computations(self, filepaths, info):
        agg_computations = pd.DataFrame(index=filepaths)
        for key in info.keys():
            if key in IMAGE_PROPERTY_ISSUE_TYPES_LIST + ["statistics"]:
                for col in info[key]:
                    agg_computations = agg_computations.join(info[key][col])
        return agg_computations

    def aggregate(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        agg_computations = pd.DataFrame(results)
        agg_computations = agg_computations.set_index("path")
        return agg_computations

    def update_info(self, agg_computations: pd.DataFrame) -> None:
        property_names = {
            issue_type: self.image_properties[issue_type].name
            for issue_type in self.issue_types
        }
        issue_columns = {
            issue_type: [
                col
                for col in agg_computations.columns
                if col.startswith(property_names[issue_type] + "_")
            ]
            for issue_type in self.issue_types
        }

        for issue_type in self.issue_types:
            self.info["statistics"][property_names[issue_type]] = agg_computations[
                property_names[issue_type]
            ]
            self.info[issue_type] = (
                agg_computations[issue_columns[issue_type]]
                if len(issue_columns[issue_type]) > 0
                else {}
            )

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
