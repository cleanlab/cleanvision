import hashlib
import multiprocessing
from typing import Any, Dict, List, Optional, Union

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from cleanvision.dataset.base_dataset import Dataset
from cleanvision.issue_managers import register_issue_manager, IssueType
from cleanvision.utils.base_issue_manager import IssueManager
from cleanvision.utils.constants import SETS, DUPLICATE, MAX_PROCS
from cleanvision.utils.utils import get_is_issue_colname, get_score_colname


def get_hash(image: Image.Image, params: Dict[str, Any]) -> str:
    hash_type, hash_size = params["hash_type"], params.get("hash_size", None)
    if hash_type == "md5":
        pixels = np.asarray(image)
        return hashlib.md5(pixels.tobytes()).hexdigest()
    elif hash_type == "whash":
        return str(imagehash.whash(image, hash_size=hash_size))
    elif hash_type == "phash":
        return str(imagehash.phash(image, hash_size=hash_size))
    elif hash_type == "ahash":
        return str(imagehash.average_hash(image, hash_size=hash_size))
    elif hash_type == "dhash":
        return str(imagehash.dhash(image, hash_size=hash_size))
    elif hash_type == "chash":
        return str(imagehash.colorhash(image, binbits=hash_size))
    else:
        raise ValueError("Hash type not supported")


def compute_hash(
    index: int,
    dataset: Dataset,
    to_compute: List[str],
    params: Dict[str, Any],
) -> Dict[str, Union[str, int]]:
    image = dataset[index]
    result: Dict[str, Union[str, int]] = {"index": index}
    if image:
        for issue_type in to_compute:
            result[issue_type] = get_hash(image, params[issue_type])
    return result


def compute_hash_wrapper(args: Dict[str, Any]) -> Dict[str, Union[str, int]]:
    return compute_hash(**args)


@register_issue_manager(DUPLICATE)
class DuplicateIssueManager(IssueManager):
    issue_name: str = DUPLICATE
    visualization: str = "image_sets"

    def __init__(self) -> None:
        super().__init__()
        self.issue_types: List[str] = []
        self.params = self.get_default_params()

    def get_default_params(self) -> Dict[str, Any]:
        return {
            IssueType.EXACT_DUPLICATES.value: {"hash_type": "md5"},
            IssueType.NEAR_DUPLICATES.value: {"hash_type": "phash", "hash_size": 8},
        }

    def update_params(self, params: Dict[str, Any]) -> None:
        for issue_type in self.params:
            non_none_params = {
                k: v for k, v in params.get(issue_type, {}).items() if v is not None
            }
            self.params[issue_type] = {**self.params[issue_type], **non_none_params}

    def _get_issue_types_to_compute(
        self, issue_types: List[str], imagelab_info: Dict[str, Any]
    ) -> List[str]:
        """Gets issue type for which computation needs to be done

        Only exact duplicate results are reused.
        Near duplicate hash computation is always done if specified in issue_types
        to reduce code complexity as it is parametric.

        Parameters
        ----------
        imagelab_info

        Returns
        -------
        to_compute : list
                     List of issue_types to run computation for
        """
        to_compute = []
        if SETS not in imagelab_info.get(IssueType.EXACT_DUPLICATES.value, {}):
            to_compute.append(IssueType.EXACT_DUPLICATES.value)
        if IssueType.NEAR_DUPLICATES.value in issue_types:
            to_compute.append(IssueType.NEAR_DUPLICATES.value)
        return to_compute

    def find_issues(
        self,
        *,
        params: Optional[Dict[str, Any]] = None,
        dataset: Optional[Dataset] = None,
        imagelab_info: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().find_issues(**kwargs)
        assert params is not None
        assert imagelab_info is not None
        assert dataset is not None
        assert n_jobs is not None

        self.issue_types = list(params.keys())
        self.update_params(params)

        to_compute = self._get_issue_types_to_compute(self.issue_types, imagelab_info)
        issue_type_hash_mapping: Dict[str, Any] = {
            issue_type: {} for issue_type in to_compute
        }

        results: List[Dict[str, Union[str, int]]] = []
        if n_jobs == 1:
            for idx in tqdm(
                dataset.index, leave=verbose, desc="Computing hashes", smoothing=0
            ):
                results.append(compute_hash(idx, dataset, to_compute, self.params))
        else:
            args = [
                {
                    "index": idx,
                    "dataset": dataset,
                    "to_compute": to_compute,
                    "params": self.params,
                }
                for idx in dataset.index
            ]
            chunksize = max(1, len(args) // MAX_PROCS)
            with multiprocessing.Pool(n_jobs) as p:
                results = list(
                    tqdm(
                        p.imap_unordered(
                            compute_hash_wrapper, args, chunksize=chunksize
                        ),
                        total=len(dataset),
                        leave=verbose,
                        desc="Computing hashes",
                        smoothing=0,
                    )
                )

            results = sorted(results, key=lambda r: r["index"])

        for result in results:
            for issue_type in to_compute:
                hash_str = result.get(issue_type, None)
                if hash_str in issue_type_hash_mapping[issue_type]:
                    issue_type_hash_mapping[issue_type][hash_str].append(
                        result["index"]
                    )
                else:
                    issue_type_hash_mapping[issue_type][hash_str] = [result["index"]]

        self.issues = pd.DataFrame(index=dataset.index)
        self._update_info(self.issue_types, issue_type_hash_mapping, imagelab_info)
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
        score_df = pd.DataFrame(index=self.issues.index)
        score_df[get_score_colname(IssueType.EXACT_DUPLICATES.value)] = np.ones(
            len(score_df)
        )
        score_df[get_score_colname(IssueType.NEAR_DUPLICATES.value)] = np.ones(
            len(score_df)
        )

        for issue_type in self.issue_types:
            score_col = get_score_colname(issue_type)
            for s in self.info[issue_type][SETS]:
                score = 1.0 / len(
                    s
                )  # will never be 0 because all images in this set are duplicated
                score_df.loc[s, score_col] = score

            self.issues = self.issues.join(score_df[[score_col]])
            self.issues[get_is_issue_colname(issue_type)] = [
                False if x == 1 else True for x in self.issues[score_col]
            ]

    def _update_info(
        self,
        issue_types: List[str],
        issue_type_hash_mapping: Dict[str, Any],
        imagelab_info: Dict[str, Any],
    ) -> None:
        num_sets = "num_sets"
        if IssueType.EXACT_DUPLICATES.value in issue_type_hash_mapping:
            self.info[IssueType.EXACT_DUPLICATES.value] = {
                SETS: self._get_duplicate_sets(
                    issue_type_hash_mapping[IssueType.EXACT_DUPLICATES.value]
                )
            }
        else:
            self.info[IssueType.EXACT_DUPLICATES.value] = {
                SETS: imagelab_info[IssueType.EXACT_DUPLICATES.value][SETS]
            }
        self.info[IssueType.EXACT_DUPLICATES.value][num_sets] = len(
            self.info[IssueType.EXACT_DUPLICATES.value][SETS]
        )

        if IssueType.NEAR_DUPLICATES.value in issue_types:
            self.info[IssueType.NEAR_DUPLICATES.value] = {
                SETS: self._get_duplicate_sets(
                    issue_type_hash_mapping[IssueType.NEAR_DUPLICATES.value]
                )
            }
            self._remove_exact_duplicates_from_near()
            self.info[IssueType.NEAR_DUPLICATES.value][num_sets] = len(
                self.info[IssueType.NEAR_DUPLICATES.value][SETS]
            )

    def _remove_exact_duplicates_from_near(self) -> None:
        updated_sets = []
        for s in self.info[IssueType.NEAR_DUPLICATES.value][SETS]:
            if s not in self.info[IssueType.EXACT_DUPLICATES.value][SETS]:
                updated_sets.append(s)
        self.info[IssueType.NEAR_DUPLICATES.value][SETS] = updated_sets

    @staticmethod
    def _get_duplicate_sets(hash_image_mapping: Dict[str, str]) -> List[str]:
        duplicate_sets = []
        for hash, images in hash_image_mapping.items():
            if len(images) > 1:
                duplicate_sets.append(images)
        return duplicate_sets
