from typing import List

from cleanvision.issue_managers import IssueType

IMAGE_PROPERTY: str = "image_property"
DUPLICATE: str = "duplicate"

# class IssueType(Enum):
#     DARK = "dark"
#     LIGHT = "light"
#     ODD_ASPECT_RATIO = "odd_aspect_ratio"
#     LOW_INFORMATION = "low_information"
#     EXACT_DUPLICATES = "exact_duplicates"
#     NEAR_DUPLICATES = "near_duplicates"
#     BLURRY = "blurry"
#     GRAYSCALE = "grayscale"
#     ODD_SIZE = "odd_size"

IMAGE_PROPERTY_ISSUE_TYPES_LIST: List[str] = [
    IssueType.DARK.value,  # "dark",
    IssueType.LIGHT.value,  # "light",
    IssueType.ODD_ASPECT_RATIO.value,  # "odd_aspect_ratio",
    IssueType.LOW_INFORMATION.value,  # "low_information",
    IssueType.BLURRY.value,  # "blurry",
    IssueType.GRAYSCALE.value,  # "grayscale",
    IssueType.ODD_SIZE.value,  # "odd_size",
]
DUPLICATE_ISSUE_TYPES_LIST: List[str] = [
    IssueType.EXACT_DUPLICATES.value,  # "exact_duplicates"
    IssueType.NEAR_DUPLICATES.value,  #  "near_duplicates"
]
SETS: str = "sets"

# max number of processes that can be forked/spawned for multiprocessing
MAX_PROCS = 5000
MAX_RESOLUTION_FOR_BLURRY_DETECTION = 64

IMAGE_FILE_EXTENSIONS: List[str] = [
    "*.jpg",
    "*.JPG",
    "*.jpeg",
    "*.JPEG",
    "*.gif",
    "*.GIF",
    "*.jp2",
    "*.JP2",
    "*.png",
    "*.PNG",
    "*.tiff",
    "*.TIFF",
    "*.webp",
    "*.WebP",
    "*.WEBP",
]  # filetypes supported by PIL

DEFAULT_ISSUE_TYPES = [
    IssueType.DARK.value,  # "dark",
    IssueType.LIGHT.value,  # "light",
    IssueType.ODD_ASPECT_RATIO.value,  # "odd_aspect_ratio",
    IssueType.LOW_INFORMATION.value,  # "low_information",
    IssueType.EXACT_DUPLICATES.value,  # "exact_duplicates"
    IssueType.NEAR_DUPLICATES.value,  # "near_duplicates"
    IssueType.BLURRY.value,  # "blurry",
    IssueType.GRAYSCALE.value,  # "grayscale",
    IssueType.ODD_SIZE.value,  # "odd_size",
    # "dark",
    # "light",
    # "odd_aspect_ratio",
    # "low_information",
    # "exact_duplicates",
    # "near_duplicates",
    # "blurry",
    # "grayscale",
    # "odd_size",
]
