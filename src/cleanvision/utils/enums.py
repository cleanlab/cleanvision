from enum import Enum


class IssueType(str, Enum):
    DARK = "dark"
    LIGHT = "light"
    ODD_ASPECT_RATIO = "odd_aspect_ratio"
    LOW_INFORMATION = "low_information"
    EXACT_DUPLICATES = "exact_duplicates"
    NEAR_DUPLICATES = "near_duplicates"
    BLURRY = "blurry"
    GRAYSCALE = "grayscale"
    ODD_SIZE = "odd_size"


class HashType(str, Enum):
    MD5 = "md5"
    WHASH = "whash"
    PHASH = "phash"
    AHASH = "ahash"
    DHASH = "dhash"
    CHASH = "chash"
