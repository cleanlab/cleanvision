from typing import List

from cleanvision.utils.enums import IssueType


IMAGE_PROPERTY: str = "image_property"
DUPLICATE: str = "duplicate"


IMAGE_PROPERTY_ISSUE_TYPES_LIST: List[str] = [
    IssueType.DARK.value,
    IssueType.LIGHT.value,
    IssueType.ODD_ASPECT_RATIO.value,
    IssueType.LOW_INFORMATION.value,
    IssueType.BLURRY.value,
    IssueType.GRAYSCALE.value,
    IssueType.ODD_SIZE.value,
]

DUPLICATE_ISSUE_TYPES_LIST: List[str] = [
    IssueType.EXACT_DUPLICATES.value,
    IssueType.NEAR_DUPLICATES.value,
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
    IssueType.DARK.value,
    IssueType.LIGHT.value,
    IssueType.ODD_ASPECT_RATIO.value,
    IssueType.LOW_INFORMATION.value,
    IssueType.EXACT_DUPLICATES.value,
    IssueType.NEAR_DUPLICATES.value,
    IssueType.BLURRY.value,
    IssueType.GRAYSCALE.value,
    IssueType.ODD_SIZE.value,
]
