from abc import ABC, abstractmethod

import pandas as pd


class IssueManager(ABC):
    """Base class for managing data issues of a particular type in Imagelab."""

    def __init__(self, params):
        self.info = {"statistics": {}}
        self.issues = pd.DataFrame()
        self.summary = pd.DataFrame(columns=["issue_type"])
        self.set_params(params)

    @property
    @abstractmethod
    def issue_name(self) -> str:
        """Returns a name that identifies the type of issue that the manager handles."""
        raise NotImplementedError

    def __repr__(self):
        class_name = self.__class__.__name__
        return class_name

    def __str__(self):
        class_name = self.__class__.__name__
        return class_name

    @abstractmethod
    def find_issues(self, /, *args, **kwargs):
        """Finds occurrences of this particular issue in the dataset."""
        raise NotImplementedError

    @abstractmethod
    def set_params(self, params):
        raise NotImplementedError

    @staticmethod
    def _compute_summary(issues_boolean):
        return {"num_images": issues_boolean.sum()}
