import math
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from PIL import ImageStat, ImageFilter

from cleanvision.issue_managers import IssueType


class ImageProperty(ABC):
    @abstractmethod
    def calculate(self, image):
        raise NotImplementedError

    @abstractmethod
    def get_scores(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def mark_issue(scores, threshold):
        return scores < threshold

def calc_brightness(image):
    stat = ImageStat.Stat(image)
    try:
        red, green, blue = stat.mean
    except ValueError:
        red, green, blue = (
            stat.mean[0],
            stat.mean[0],
            stat.mean[0],
        )  # deals with black and white images

    cur_bright = (
        math.sqrt(0.241 * (red**2) + 0.691 * (green**2) + 0.068 * (blue**2))
    ) / 255
    return cur_bright

class BrightnessProperty(ImageProperty):
    name = "brightness"

    def __init__(self, issue_type):
        self.issue_type = issue_type

    def calculate(self, image):
        return calc_brightness(image)

    def get_scores(self, raw_scores, **_):
        scores = np.array(raw_scores)
        scores[scores > 1] = 1
        # reverse the brightness scores to catch images which are too bright
        if self.issue_type.name == IssueType.LIGHT.name:
            scores = 1 - scores
        return scores


def calc_aspect_ratio(image):
    width, height = image.size
    size_score = min(width / height, height / width)  # consider extreme shapes
    return size_score


class AspectRatioProperty(ImageProperty):
    name = "aspect_ratio"

    def calculate(self, image):
        return calc_aspect_ratio(image)

    def get_scores(self, raw_scores, **_):
        scores = np.array(raw_scores)
        return scores


def calc_entropy(image):
    return image.entropy()

class EntropyProperty(ImageProperty):
    name = "entropy"

    def calculate(self, image):
        return calc_entropy(image)

    def get_scores(self, raw_scores, normalizing_factor, **_):
        scores = np.array(raw_scores)
        scores = normalizing_factor * scores
        scores[scores > 1] = 1
        return scores


def calc_blurriness(image):
        edges = get_edges(image)
        return ImageStat.Stat(edges).var[0]

class BlurrinessProperty(ImageProperty):
    name = "blurriness"

    def calculate(self, image):
        return calc_blurriness(image)

    def get_scores(self, raw_scores, normalizing_factor, **_):
        raw_scores = np.array(raw_scores)
        scores = 1 - np.exp(-1 * raw_scores * normalizing_factor)
        return scores


def get_edges(image):
    gray_image = image.convert("L")
    edges = gray_image.filter(ImageFilter.FIND_EDGES)
    return edges


def calculate_brightness(red, green, blue):
    cur_bright = (
        math.sqrt(0.241 * (red**2) + 0.691 * (green**2) + 0.068 * (blue**2))
    ) / 255

    return cur_bright


def calc_color_space(image):
    return get_image_mode(image)


class ColorSpaceProperty(ImageProperty):
    name = "color_space"

    def calculate(self, image):
        return calc_color_space(image)

    def get_scores(self, raw_scores: List[str], **_):
        scores = np.array([0 if mode == "L" else 1 for mode in raw_scores])
        return scores

    def mark_issue(self, scores, *_):
        return (1 - scores).astype("bool")


def get_image_mode(image):
    if image.mode:
        return image.mode
    else:
        imarr = np.asarray(image)
        if len(imarr.shape) == 2 or (
            len(imarr.shape) == 3
            and (np.diff(imarr.reshape(-1, 3).T, axis=0) == 0).all()
        ):
            return "L"
        else:
            return "UNK"
