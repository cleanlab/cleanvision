import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from clean_vision.issue_managers.base_issue_manager import IssueManager
from clean_vision.utils.issue_types import IssueType


class CustomIssueManager(IssueManager):
    def __init__(self):
        super().__init__()

    @classmethod
    @property
    def issue_name(cls):
        return IssueType.CUSTOM_IMAGES

    @staticmethod
    def calculate_mean_pixel_value(image):
        gray_image = image.convert("L")
        return np.mean(np.array(gray_image))

    def get_scores(self, raw_scores):
        scores = np.array(raw_scores)
        return scores / 255.0

    def mark_issue(self, scores, threshold):
        return scores < threshold

    def update_summary(self, summary_dict):
        self.summary = pd.DataFrame({"issue_type": [self.issue_name.value]})
        for column_name, value in summary_dict.items():
            self.summary[column_name] = [value]

    def find_issues(self, filepaths, imagelab_info):
        raw_scores = []
        for path in tqdm(filepaths):
            image = Image.open(path)
            raw_scores.append(self.calculate_mean_pixel_value(image))

        self.issues = pd.DataFrame(index=filepaths)
        scores = self.get_scores(raw_scores)
        self.issues[f"{self.issue_name}_score"] = scores
        self.issues[f"{self.issue_name}_bool"] = self.mark_issue(
            scores, self.issue_name.threshold
        )
        self.info["PixelValue"] = raw_scores
        summary_dict = self._compute_summary(self.issues[f"{self.issue_name}_bool"])

        self.update_summary(summary_dict)
