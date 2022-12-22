from abc import ABC, abstractmethod

import pandas as pd


class IssueManager(ABC):
    """Base class for managing data issues of a particular type in Imagelab."""

    def __init__(self):
        self.info = {}
        self.issues = pd.DataFrame(columns=["image_path"])
        self.summary = pd.DataFrame(columns=["issue_type", "num_images"])

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

    @staticmethod
    def _compute_summary(issues_boolean):
        return {"num_images": issues_boolean.sum()}
