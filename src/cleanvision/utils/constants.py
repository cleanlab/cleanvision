from typing import List

IMAGE_PROPERTY: str = "image_property"
DUPLICATE: str = "duplicate"
IMAGE_PROPERTY_ISSUE_TYPES_LIST: List[str] = [
    "dark",
    "light",
    "odd_aspect_ratio",
    "low_information",
    "blurry",
    "grayscale",
]
DUPLICATE_ISSUE_TYPES_LIST: List[str] = ["exact_duplicates", "near_duplicates"]
SETS: str = "sets"

# max number of processes that can be forked/spawned for multiprocessing
MAX_PROCS = 5000
MAX_RESOLUTION_FOR_BLURRY_DETECTION = 256
