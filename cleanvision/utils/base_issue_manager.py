from abc import ABC, abstractmethod
from typing import Dict, TypeVar, Any, List, Type

import pandas as pd

TIssueManager = TypeVar(
    "TIssueManager", bound="IssueManager"
)  # self type for the class


class IssueManager(ABC):
    """Base class for managing data issues of a particular type in Imagelab."""

    def __init__(self: TIssueManager, params: Dict[str, Any]):
        self.info: Dict[str, Dict[str, Any]] = {"statistics": {}}
        self.issues: pd.DataFrame = pd.DataFrame()
        self.summary: pd.DataFrame = pd.DataFrame(columns=["issue_type"])
        self.params: Dict[str, Any] = self.get_default_params()
        self.set_params(params)

    def __repr__(self: TIssueManager) -> str:
        class_name = self.__class__.__name__
        return class_name

    def __str__(self: TIssueManager) -> str:
        class_name = self.__class__.__name__
        return class_name

    @staticmethod
    def check_params(**kwargs: Any) -> None:
        allowed_kwargs: Dict[str, Any] = {
            "filepaths": List[str],
            "imagelab_info": Dict[str, Any],
        }

        for name, value in kwargs.items():
            if name not in allowed_kwargs:
                raise ValueError(f"{name} is not a valid keyword argument.")
            if not isinstance(value, allowed_kwargs[name]):
                raise ValueError(
                    f"Valid type for keyword argument {name} can only be {allowed_kwargs[name]}. {name} cannot be type {type(name)}. "
                )

    @abstractmethod
    def find_issues(self: TIssueManager, **kwargs: Any) -> None:
        """Finds occurrences of this particular issue in the dataset."""
        self.check_params(**kwargs)
        return

    @abstractmethod
    def get_default_params(self: TIssueManager) -> Dict[str, Any]:
        """Returns default params to be used by the issue_manager"""
        raise NotImplementedError

    @abstractmethod
    def set_params(self, params: Dict[str, Any]) -> None:
        """Sets params for an issue manager. Default params will be overridden by user provided params"""
        raise NotImplementedError

    @staticmethod
    def _compute_summary(issues_boolean: "pd.Series[bool]") -> Dict[str, int]:
        return {"num_images": issues_boolean.sum()}
