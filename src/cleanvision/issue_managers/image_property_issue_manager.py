import multiprocessing
from typing import Dict, Any, List, Set, Optional, Union

import pandas as pd
from tqdm import tqdm

from cleanvision.dataset.base_dataset import Dataset
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
from cleanvision.utils.utils import (
    get_is_issue_colname,
    update_df,
    get_score_colname,
)


def compute_scores(
    index: int,
    dataset: Dataset,
    to_compute: List[str],
    image_properties: Dict[str, ImageProperty],
) -> Dict[str, Union[str, int, float]]:
    image = dataset[index]
    result: Dict[str, Union[int, str, float]] = {"index": index}
    for issue_type in to_compute:
        result = {**result, **image_properties[issue_type].calculate(image)}
    return result


def compute_scores_wrapper(args: Dict[str, Any]) -> Dict[str, Union[float, str, int]]:
    return compute_scores(**args)


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
            IssueType.BLURRY.value: {"threshold": 0.17, "normalizing_factor": 0.01},
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

    def _get_additional_to_compute_set(self, issue_types: List[str]) -> List[str]:
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
        dataset: Optional[Dataset] = None,
        imagelab_info: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().find_issues(**kwargs)
        assert params is not None
        assert imagelab_info is not None
        assert dataset is not None
        assert n_jobs is not None

        self.issue_types = list(params.keys())
        self.issues = pd.DataFrame(index=dataset.index)
        additional_set = self._get_additional_to_compute_set(self.issue_types)
        self.issue_types = self.issue_types + additional_set

        self.update_params(params)

        agg_computations = pd.DataFrame(index=dataset.index)
        agg_computations = self._add_prev_computations(agg_computations, imagelab_info)

        defer_set = self._get_defer_set(self.issue_types, agg_computations)

        to_be_computed = list(set(self.issue_types).difference(defer_set))

        new_computations = pd.DataFrame(index=dataset.index)
        if to_be_computed:
            results: List[Dict[str, Union[int, float, str]]] = []
            if n_jobs == 1:
                for idx in tqdm(dataset.index):
                    results.append(
                        compute_scores(
                            idx, dataset, to_be_computed, self.image_properties
                        )
                    )
            else:
                args = [
                    {
                        "index": idx,
                        "dataset": dataset,
                        "to_compute": to_be_computed,
                        "image_properties": self.image_properties,
                    }
                    for idx in dataset.index
                ]
                chunksize = max(1, len(args) // MAX_PROCS)
                with multiprocessing.Pool(n_jobs) as p:
                    results = list(
                        tqdm(
                            p.imap_unordered(
                                compute_scores_wrapper, args, chunksize=chunksize
                            ),
                            total=len(dataset),
                        )
                    )

                results = sorted(results, key=lambda r: r["index"])

            new_computations = self._aggregate(results)

        agg_computations = update_df(agg_computations, new_computations)

        ordered_issue_types = self._get_dependency_sorted(self.issue_types)

        self.update_issues(agg_computations, ordered_issue_types)
        self.update_info(agg_computations)
        self.update_summary()
        return

    def _get_dependency_sorted(self, issue_types: List[str]) -> List[str]:
        # todo: remove this hacky way, add an ordering
        return sorted(issue_types, reverse=True)

    def update_issues(
        self, agg_computations: pd.DataFrame, issue_types: List[str]
    ) -> None:
        """Updates `self.issues` with score and is_issue columns

        Parameters
        ----------
        agg_computations: pd.DataFrame
            This dataframe contains all computed properties like blurriness, brightness as columns for each image
            that are required for computing issue scores.
        issue_types: List[str]
            List of issue types for which to update `self.issues`
        """
        for issue_type in issue_types:
            score_column_names = self.image_properties[issue_type].score_columns
            score_columns = agg_computations[score_column_names]

            # todo: this is hacky
            # Only blurry issue is dependent on another issue (in this case dark) for computing scores.
            # This if else block handles this special case.
            if issue_type == IssueType.BLURRY.value:
                # In the case when blurry scores need to be computed
                # dark_score and is_dark_issue can be retrieved from one of these two places
                # 1. self.issues
                # 2. recomputed using brightness_perc_99 info present in agg_computations.
                dark_issue = IssueType.DARK.value
                if not {
                    get_is_issue_colname(dark_issue),
                    get_score_colname(dark_issue),
                }.issubset(self.issues):
                    dark_score_columns = agg_computations[
                        self.image_properties[dark_issue].score_columns
                    ]
                    dark_property = self.image_properties[dark_issue]

                    dark_issue_scores = dark_property.get_scores(
                        dark_score_columns, dark_issue, **self.params[dark_issue]
                    )
                    is_dark_issue = dark_property.mark_issue(
                        dark_issue_scores,
                        self.params[dark_issue].get("threshold"),
                        dark_issue,
                    )
                else:
                    dark_issue_scores = self.issues[[get_score_colname(dark_issue)]]
                    is_dark_issue = self.issues[[get_is_issue_colname(dark_issue)]]

                issue_scores = self.image_properties[issue_type].get_scores(
                    score_columns,
                    issue_type,
                    **self.params[issue_type],
                    dark_issue_data=dark_issue_scores.join(is_dark_issue),
                )
            else:
                issue_scores = self.image_properties[issue_type].get_scores(
                    score_columns, issue_type, **self.params[issue_type]
                )

            is_issue = self.image_properties[issue_type].mark_issue(
                issue_scores, self.params[issue_type].get("threshold"), issue_type
            )
            self.issues = self.issues.join(issue_scores)
            self.issues = self.issues.join(is_issue)

    def _add_prev_computations(
        self, agg_computations: pd.DataFrame, info: Dict[str, Any]
    ) -> pd.DataFrame:
        for key in info.keys():
            if key in IMAGE_PROPERTY_ISSUE_TYPES_LIST + ["statistics"]:
                for col in info[key]:
                    if col not in agg_computations:
                        agg_computations = agg_computations.join(info[key][col])
        return agg_computations

    def _aggregate(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        agg_computations = pd.DataFrame(results)
        agg_computations = agg_computations.set_index("index")
        return agg_computations

    def update_info(self, agg_computations: pd.DataFrame) -> None:
        """Updates `self.info` using computed properties

        Parameters
        ----------
        agg_computations: pd.DataFrame
            This dataframe contains all computed properties like blurriness, brightness as columns for each image
            that are required for computing issue scores.
        """
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
