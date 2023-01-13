import math
from abc import ABC, abstractmethod

import numpy as np
from PIL import ImageStat

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


class BrightnessProperty(ImageProperty):
    name = "Brightness"

    def __init__(self, issue_type):
        self.issue_type = issue_type

    def calculate(self, image):
        stat = ImageStat.Stat(image)
        try:
            red, green, blue = stat.mean
        except ValueError:
            red, green, blue = (
                stat.mean[0],
                stat.mean[0],
                stat.mean[0],
            )  # deals with black and white images

        cur_bright = calculate_brightness(red, green, blue)

        return cur_bright

    def get_scores(self, raw_scores, **_):
        scores = np.array(raw_scores)
        scores[scores > 1] = 1
        # reverse the brightness scores to catch images which are too bright
        if self.issue_type.name == IssueType.LIGHT.name:
            scores = 1 - scores
        return scores


class AspectRatioProperty(ImageProperty):
    name = "AspectRatio"

    def calculate(self, image):
        width, height = image.size
        size_score = min(width / height, height / width)  # consider extreme shapes
        return size_score

    def get_scores(self, raw_scores, **_):
        scores = np.array(raw_scores)
        return scores


class EntropyProperty(ImageProperty):
    name = "Entropy"

    def calculate(self, image):
        entropy = image.entropy()
        return entropy

    def get_scores(self, raw_scores, normalizing_factor, **_):
        scores = np.array(raw_scores)
        scores = normalizing_factor * scores
        scores[scores > 1] = 1
        return scores


def calculate_brightness(red, green, blue):
    cur_bright = (
        math.sqrt(0.241 * (red**2) + 0.691 * (green**2) + 0.068 * (blue**2))
    ) / 255

    return cur_bright
