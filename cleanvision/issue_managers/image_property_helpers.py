import math
from abc import ABC, abstractmethod

import numpy as np
from PIL import ImageStat

from cleanvision.issue_managers import IssueType


class ImagePropertyHelper(ABC):
    @abstractmethod
    def calculate(self, image):
        raise NotImplementedError

    @abstractmethod
    def normalize(self, raw_scores):
        raise NotImplementedError

    @staticmethod
    def mark_issue(scores, threshold):
        return scores < threshold


class BrightnessHelper(ImagePropertyHelper):
    image_property = "Brightness"

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

    def normalize(self, raw_scores):
        scores = np.array(raw_scores)
        scores[scores > 1] = 1
        # reverse the brightness scores to catch images which are too bright
        if self.issue_type.name == IssueType.LIGHT.name:
            scores = 1 - scores
        return scores


def calculate_brightness(red, green, blue):

    cur_bright = (
        math.sqrt(0.241 * (red**2) + 0.691 * (green**2) + 0.068 * (blue**2))
    ) / 255

    return cur_bright
