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

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(params)
        self.issue_types: List[str] = list(params.keys())

    def get_default_params(self) -> Dict[str, Any]:
        return {
            IssueType.EXACT_DUPLICATES.value: {"hash_type": "md5"},
            IssueType.NEAR_DUPLICATES.value: {"hash_type": "whash", "hash_size": 8},
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        updated_params = {}
        for issue_type in self.params:
            non_none_params = {
                k: v for k, v in params.get(issue_type, {}).items() if v is not None
            }
            updated_params[issue_type] = {**self.params[issue_type], **non_none_params}
        self.params = updated_params

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
        self, imagelab_info: Dict[str, Any]
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
        if IssueType.NEAR_DUPLICATES.value in self.issue_types:
            to_compute.append(IssueType.NEAR_DUPLICATES.value)
        return to_compute

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

        to_compute = self._get_issue_types_to_compute(imagelab_info)
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
        self._update_info(issue_type_hash_mapping, imagelab_info)
        self._update_issues()
        self._update_summary()

        return

    def _update_summary(self) -> None:
        summary_dict = {}
        for issue_type in self.issue_types:
            issues_boolean = self.issues[f"{issue_type}_bool"]
            summary_dict[issue_type] = self._compute_summary(issue_type)
        summary_df = pd.DataFrame.from_dict(summary_dict, orient="index")
        summary_df["issue_type"] = summary_df.index
        self.summary = summary_df.reset_index()

    def _compute_summary(
        self, issue_type: str
    ) -> Dict[str, Any]:
        return {
            "num_images": self.issues[f"{issue_type}_bool"].sum(),
            "num_sets": len(self.info[issue_type][SETS]),
        }

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
        issue_type_hash_mapping: Dict[str, Any],
        imagelab_info: Dict[str, Any],
    ) -> None:
        if IssueType.EXACT_DUPLICATES.value in imagelab_info:
            self.info[IssueType.EXACT_DUPLICATES.value] = {
                SETS: imagelab_info[IssueType.EXACT_DUPLICATES.value][SETS]
            }

        for issue_type in issue_type_hash_mapping:
            self.info[issue_type] = {
                SETS: self._get_duplicate_sets(issue_type_hash_mapping[issue_type])
            }
        self._remove_exact_duplicates_from_near()

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
