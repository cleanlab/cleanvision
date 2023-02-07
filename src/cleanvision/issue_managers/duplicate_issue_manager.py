import hashlib
from typing import Any, Dict, List, Optional

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm

from cleanvision.issue_managers import register_issue_manager, IssueType
from cleanvision.utils.base_issue_manager import IssueManager
from cleanvision.utils.constants import SETS, DUPLICATE


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
            IssueType.NEAR_DUPLICATES.value: {"hash_type": "whash", "hash_size": 8},
        }

    def update_params(self, params: Dict[str, Any]) -> None:
        self.params = self.get_default_params()
        for issue_type in self.params:
            non_none_params = {
                k: v for k, v in params.get(issue_type, {}).items() if v is not None
            }
            self.params[issue_type] = {**self.params[issue_type], **non_none_params}

    @staticmethod
    def _get_hash(image: Image.Image, params: Dict[str, Any]) -> str:
        hash_type, hash_size = params["hash_type"], params.get("hash_size", None)
        if hash_type == "md5":
            return hashlib.md5(image.tobytes()).hexdigest()
        elif hash_type == "whash":
            return str(imagehash.whash(image, hash_size=hash_size))
        elif hash_type == "phash":
            return str(imagehash.phash(image, hash_size=hash_size))
        else:
            raise ValueError("Hash type not supported")

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
        filepaths: Optional[List[str]] = None,
        imagelab_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().find_issues(**kwargs)
        assert params is not None
        assert imagelab_info is not None
        assert filepaths is not None

        self.issue_types = list(params.keys())
        self.update_params(params)

        to_compute = self._get_issue_types_to_compute(self.issue_types, imagelab_info)
        issue_type_hash_mapping: Dict[str, Any] = {
            issue_type: {} for issue_type in to_compute
        }

        for path in tqdm(filepaths):
            image = Image.open(path)
            for issue_type in to_compute:
                hash = self._get_hash(image, self.params[issue_type])
                if hash in issue_type_hash_mapping[issue_type]:
                    issue_type_hash_mapping[issue_type][hash].append(path)
                else:
                    issue_type_hash_mapping[issue_type][hash] = [path]

        self.issues = pd.DataFrame(index=filepaths)
        self._update_info(self.issue_types, issue_type_hash_mapping, imagelab_info)
        self._update_issues()
        self._update_summary()

        return

    def _update_summary(self) -> None:
        summary_dict = {}
        for issue_type in self.issue_types:
            summary_dict[issue_type] = self._compute_summary(
                self.issues[f"{issue_type}_bool"]
            )
        summary_df = pd.DataFrame.from_dict(summary_dict, orient="index")
        self.summary = summary_df.reset_index()
        self.summary = self.summary.rename(columns={"index": "issue_type"})
        self.summary = self.summary.astype({"issue_type": str})

    def _update_issues(self) -> None:
        for issue_type in self.issue_types:
            duplicated_images = []
            for s in self.info[issue_type][SETS]:
                duplicated_images.extend(s)
            self.issues[f"{issue_type}_bool"] = self.issues.index.to_series().apply(
                lambda x: True if x in duplicated_images else False
            )

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
