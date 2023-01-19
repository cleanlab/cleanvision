from abc import ABC, abstractmethod
from typing import Dict, TypeVar, Any, List, Type

import pandas as pd

TIssueManager = TypeVar("TIssueManager", bound="IssueManager")  # self type for the class


class IssueManager(ABC):
    """Base class for managing data issues of a particular type in Imagelab."""

    def __init__(self: TIssueManager, params: Dict[Any, Any]):
        self.info: Dict[str, Dict[Any, Any]] = {"statistics": {}}
        self.issues: pd.DataFrame = pd.DataFrame()
        self.summary: pd.DataFrame = pd.DataFrame(columns=["issue_type"])
        self.params: Dict[Any, Any] = self.get_default_params()
        self.set_params(params)

    def __repr__(self: TIssueManager) -> str:
        class_name = self.__class__.__name__
        return class_name

    def __str__(self: TIssueManager) -> str:
        class_name = self.__class__.__name__
        return class_name

    @abstractmethod
    def find_issues(self: TIssueManager, *args: List[str], **kwargs: List[str]) -> None:
        """Finds occurrences of this particular issue in the dataset."""
        raise NotImplementedError

    @abstractmethod
    def get_default_params(self: TIssueManager) -> Dict[Any, Any]:
        """Returns default params to be used by the issue_manager"""
        raise NotImplementedError

    @abstractmethod
    def set_params(self, params: Dict[Any, Any]) -> None:
        """Sets params for an issue manager. Default params will be overridden by user provided params"""
        raise NotImplementedError

    @staticmethod
    def _compute_summary(issues_boolean: "pd.Series[int]") -> Dict[str, int]:
        return {"num_images": issues_boolean.sum()}
