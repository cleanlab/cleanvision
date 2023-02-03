from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from cleanvision.issue_managers import register_issue_manager
from cleanvision.utils.base_issue_manager import IssueManager

ISSUE_NAME = "custom"


@register_issue_manager(ISSUE_NAME)
class CustomIssueManager(IssueManager):
    """
    Example class showing how you can self-define a custom type of issue that
    CleanVision can simultaneously check your data for alongside its built-in issue types.
    """

    issue_name: str = ISSUE_NAME
    visualization: str = "property_based"

    def __init__(self) -> None:
        super().__init__()
        self.params = self.get_default_params()

    def get_default_params(self) -> Dict[str, Any]:
        return {"threshold": 0.4}

    def update_params(self, params: Dict[str, Any]) -> None:
        self.params = self.get_default_params()
        non_none_params = {k: v for k, v in params.items() if v is not None}
        self.params = {**self.params, **non_none_params}

    @staticmethod
    def calculate_mean_pixel_value(image: Image.Image) -> float:
        gray_image = image.convert("L")
        return np.mean(np.array(gray_image))

    def get_scores(self, raw_scores: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
        scores = np.array(raw_scores)
        return scores / 255.0

    def mark_issue(
        self, scores: "np.ndarray[Any, Any]", threshold: float
    ) -> "np.ndarray[Any, Any]":
        return scores < threshold

    def update_summary(self, summary_dict: Dict[str, Any]) -> None:
        self.summary = pd.DataFrame({"issue_type": [self.issue_name]})
        for column_name, value in summary_dict.items():
            self.summary[column_name] = [value]

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

        self.update_params(params)

        raw_scores = []
        for path in tqdm(filepaths):
            image = Image.open(path)
            raw_scores.append(self.calculate_mean_pixel_value(image))

        self.issues = pd.DataFrame(index=filepaths)
        scores = self.get_scores(raw_scores)
        self.issues[f"{self.issue_name}_score"] = scores
        self.issues[f"{self.issue_name}_bool"] = self.mark_issue(
            scores, self.params["threshold"]
        )
        self.info[self.issue_name] = {"PixelValue": raw_scores}
        summary_dict = self._compute_summary(self.issues[f"{self.issue_name}_bool"])

        self.update_summary(summary_dict)
