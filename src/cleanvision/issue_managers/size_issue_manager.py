import multiprocessing
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from cleanvision.dataset.base_dataset import Dataset
from cleanvision.issue_managers import register_issue_manager, IssueType
from cleanvision.utils.base_issue_manager import IssueManager
from cleanvision.utils.constants import SIZE, MAX_PROCS
from cleanvision.utils.utils import get_is_issue_colname, get_score_colname
from cleanvision.utils.constants import SIZE_ISSUE_TYPES_LIST


def get_size(
    index: int,
    dataset: Dataset,
) -> Dict[str, Union[str, int]]:
    image = dataset[index]
    result: Dict[str, Union[str, List[int]]] = {"index": index}
    size = image.size
    result["width"] = size[0]
    result["height"] = size[1]
    return result


def get_size_wrapper(args: Dict[str, Any]) -> Dict[str, Union[str, int]]:
    return get_size(**args)


@register_issue_manager(SIZE)
class SizeIssueManager(IssueManager):
    issue_name: str = SIZE
    visualization: str = "individual_images"

    def __init__(self) -> None:
        super().__init__()
        self.issue_types: List[str] = []
        self.params = self.get_default_params()
        self.means = {}

    def get_default_params(self) -> Dict[str, Dict[str, float]]:
        return {IssueType.SIZE.value: {"ratio": 5.0}}

    def update_params(self, params: Dict[str, float]) -> None:
        if params:
            self.params[SIZE] = params

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

        self.issue_types = SIZE_ISSUE_TYPES_LIST
        self.update_params(params)

        results: List[Dict[str, Union[str, int]]] = []
        if n_jobs == 1:
            for idx in tqdm(dataset.index):
                results.append(get_size(idx, dataset))
        else:
            args = [
                {
                    "index": idx,
                    "dataset": dataset,
                }
                for idx in dataset.index
            ]
            chunksize = max(1, len(args) // MAX_PROCS)
            with multiprocessing.Pool(n_jobs) as p:
                results = list(
                    tqdm(
                        p.imap_unordered(get_size_wrapper, args, chunksize=chunksize),
                        total=len(dataset),
                    )
                )

        self.issues = pd.DataFrame.from_records(results)
        self.issues.set_index("index", inplace=True)

        for issue_type in self.issue_types:
            self.means[issue_type] = self.issues[issue_type].mean()

        self._update_issues()
        self._update_summary()

        return

    def _update_summary(self) -> None:
        summary_dict = {}
        for issue_type in self.issue_types:
            summary_dict[issue_type] = self._compute_summary(
                self.issues[get_is_issue_colname(issue_type)]
            )
        summary_df = pd.DataFrame.from_dict(summary_dict, orient="index")
        self.summary = summary_df.reset_index()
        self.summary = self.summary.rename(columns={"index": "issue_type"})
        self.summary = self.summary.astype({"issue_type": str})

    def _update_issues(self) -> None:
        for issue_type in self.issue_types:
            # Calculate image to mean ratio
            self.issues[get_score_colname(issue_type) + "_raw"] = self.issues[
                issue_type
            ].apply(lambda x: x / self.means[issue_type])

            # Check where its out of the range.
            self.issues[get_is_issue_colname(issue_type)] = np.where(
                self.issues[get_score_colname(issue_type) + "_raw"]
                > self.params[SIZE]["ratio"],
                True,
                False,
            )

            # Normalize the score between 0 and 1
            self.issues[get_score_colname(issue_type)] = (
                self.issues[get_score_colname(issue_type) + "_raw"]
                / self.params[SIZE]["ratio"]
            )
