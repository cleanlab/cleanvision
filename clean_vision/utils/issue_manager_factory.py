# from typing import List, Type
#
# from clean_vision.issue_managers.base_issue_manager import IssueManager
# from clean_vision.issue_managers.custom_issue_manager import CustomIssueManager
# from clean_vision.issue_managers.image_property_issue_manager import (
#     ImagePropertyIssueManager,
# )
# from clean_vision.utils.constants import IMAGE_PROPERTY
# from clean_vision.utils.issue_typess import IssueType
#
#
# class _IssueManagerFactory:
#     """Factory class for constructing concrete issue managers."""
#
#     # todo: convert these strings to constants
#     types = {
#         IMAGE_PROPERTY: ImagePropertyIssueManager,
#         IssueType.CUSTOM_IMAGES.value: CustomIssueManager,
#     }
#
#     @classmethod
#     def from_str(cls, issue_type: str) -> Type[IssueManager]:
#         """Constructs a concrete issue manager from a string."""
#         if isinstance(issue_type, list):
#             raise ValueError(
#                 "issue_type must be a string, not a list. Try using from_list instead."
#             )
#         if issue_type not in cls.types:
#             raise ValueError(f"Invalid issue type: {issue_type}")
#         return cls.types[issue_type]
#
#     @classmethod
#     def from_list(cls, issue_types: List[str]) -> List[Type[IssueManager]]:
#         """Constructs a list of concrete issue managers from a list of strings."""
#         return [cls.from_str(issue_type) for issue_type in issue_types]
