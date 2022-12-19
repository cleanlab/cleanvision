import math
from abc import ABC, abstractmethod

import numpy as np
from PIL import ImageStat

from issue_types import IssueType


class ImagePropertyHelper(ABC):
    def __init__(self):
        pass

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
    def __init__(self, issue_type):
        self.issue_type = issue_type

    def calculate(self, image):
        stat = ImageStat.Stat(image)
        try:
            r, g, b = stat.mean
        except:
            r, g, b = (
                stat.mean[0],
                stat.mean[0],
                stat.mean[0],
            )  # deals with black and white images
            # print(f"WARNING: {img} does not have just r, g, b values")
        cur_bright = (
            math.sqrt(0.241 * (r**2) + 0.691 * (g**2) + 0.068 * (b**2))
        ) / 255
        return cur_bright

    def normalize(self, raw_scores):
        scores = np.array(raw_scores)
        scores[scores > 1] = 1
        if self.issue_type.name == IssueType.WHITE_IMAGES.name:
            scores = 1 - scores
        return scores
