import importlib
import os
from enum import Enum
from typing import List, Type

from clean_vision.utils.base_issue_manager import IssueManager
from clean_vision.utils.constants import IMAGE_PROPERTY


class IssueType(Enum):
    DARK = "Dark"
    LIGHT = "Light"
    CUSTOM = "Custom"


ISSUE_MANAGER_REGISTRY = {}
ISSUE_TYPES_REGISTRY = [issue_type.value for issue_type in list(IssueType)]


class IssueManagerFactory:
    """Factory class for constructing concrete issue managers."""

    @classmethod
    def from_str(cls, issue_type: str) -> Type[IssueManager]:
        """Constructs a concrete issue manager from a string."""
        if isinstance(issue_type, list):
            raise ValueError(
                "issue_type must be a string, not a list. Try using from_list instead."
            )
        return ISSUE_MANAGER_REGISTRY[issue_type]

    @classmethod
    def from_list(cls, issue_types: List[str]) -> List[Type[IssueManager]]:
        """Constructs a list of concrete issue managers from a list of strings."""
        return [ISSUE_MANAGER_REGISTRY(issue_type) for issue_type in issue_types]


def register_issue_manager(name):
    def register_issue_manager_cls(cls):
        if name in ISSUE_MANAGER_REGISTRY:
            return ISSUE_MANAGER_REGISTRY[name]
        if not issubclass(cls, IssueManager):
            raise ValueError("{} must extend IssueManager".format(cls.__name__))

        ISSUE_MANAGER_REGISTRY[name] = cls
        if name != IMAGE_PROPERTY:
            ISSUE_TYPES_REGISTRY.append(name)
        return cls

    return register_issue_manager_cls


for file in os.listdir(os.path.dirname(__file__)):
    if (
        file.endswith("issue_manager.py")
        and not file.startswith("_")
        and not file.startswith(".")
    ):
        module_name = file[: file.find(".py")]
        module = importlib.import_module(
            "clean_vision.issue_managers" + "." + module_name
        )

# IssueType = Enum(
#     "IssueType",
#     [(issue_type.upper(), issue_type) for issue_type in ISSUE_TYPES_REGISTRY],
# )
