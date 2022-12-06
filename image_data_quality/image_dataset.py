import os
from ctypes import Union
from typing import List, Type, Any

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from PIL import Image
from tqdm import tqdm
from image_data_quality.issue_checks import (
    check_brightness,
    check_odd_size,
    check_entropy,
    check_static,
    check_blurriness,
    check_duplicated,
    check_near_duplicates,
)
from image_data_quality.utils.utils import analyze_scores_old, get_sorted_images, display_images, get_zscores, get_is_issue
from .issue_checks import check_odd_size, check_duplicated, check_near_duplicates

POSSIBLE_ISSUES = {
    "Duplicated": check_duplicated,
    "Brightness": check_brightness,
    "Odd Size": check_odd_size,
    "Blurred": check_blurriness,
    "Entropy": check_entropy,
    # "Potential Occlusion": check_entropy,
    # "Potential Static": check_static,
    "Near Duplicates": check_near_duplicates,
}

DATASET_WIDE_ISSUES = {
    "Duplicated",
    "Near Duplicates",
}  # issues requiring info. from entire dataset

# Constants:
IMAGES_DIR = "../image_files/"
ISSUES_FILENAME = "issues.csv"
# RESULTS_FILENAME = "results.pkl"
# INFO_FILENAME = "info.pkl"

class Imagelab:
    """
    An object used to load and identify different quality issues within an image dataset.

    Auto-detects issues in image data that may be problematic for machine learning
    This class takes in a dataset of images and performs targeted checks
    to report the indices of problematic images and other relevant information.

    Parameters
    ----------
    path: str, optional
      Path to folder where image dataset is located
      If not provided with path, default set to current working directory.

    image_files: list[str], optional
      A list of filenames in the image dataset sorted numerically and alphabetically
      If provided with filenames, sorts using built-in sorting function.
      Default set to list of all images in the dataset.

    thumbnail_size: tuple[int, int], optional
      A tuple specifying the size of the thumbnail in which image analysis is based on.
      If not provided with the 2-tuple, default set to (128, 128).
    """
    issue_scores = {}
    issue_info = {}
    """key: issue name string, value: list of indices of images with this issue (dict)"""
    misc_info = {}
    """key: miscellanous info name string, value: intuitive data structure containing that info (dict)"""

    def __init__(
            self,
            path: str = None,
            image_files: list = None,
            thumbnail_size: tuple = None
    ) -> None:
        if path is None:
            self.path = os.getcwd()
        else:
            self.path = path
        if image_files is None:
            self.image_files = get_sorted_images(self.path)
        else:
            self.image_files = sorted(image_files)
        if thumbnail_size is None:
            self.thumbnail_size = (128, 128)
        else:
            self.thumbnail_size = thumbnail_size
        self.verbose = None
        self.issue_managers = None
        self.issue_types = None
        self.issue_df = None
        self.issue_scores = None
        self.results = None
        self.thresholds = 5 # TODO Ulya double check

    def __repr__(self):
        """What is displayed in console if user executes: >>> imagelab"""
        if self.issue_info == {}:
            num_issues = None
        else:
            num_issues = 0
            for check in self.issue_info.values():
                if type(check[0]) == list:
                    flat_issue = []
                    for l in check: 
                        flat_issue += l
                    num_issues += len(flat_issue)
                else: 
                    num_issues += len(check) 
        display_str = "ImageDataset(num_images = " + str(len(self.image_files)) + ", path = " + str(self.path) + ", num_images_with_issue = " + str(num_issues)+")"
        # Useful info could be: num_images, path, number of images with issues identified so far (numeric or None if issue-finding not run yet).
        return display_str
  
    def __str__(self):
        """What is displayed if user executes: print(imagelab)"""
        return self.__repr__()[13:-1]  # display_info could be same information as above in display_str without the ImageDataset(...) wrapper text.   

    def get_info(self, issue_name) -> Any:
        """Returns dict of info about a specific issue, or None if this issue does not exist in self.info.
        Internally fetched from self.info[issue_name] and prettified.
        Keys might include: number of examples suffering from issue, indicates of top-K examples most severely suffering,
        other misc stuff like which sets of examples are duplicates if the issue=="duplicated".
        """
        if issue_name in self.info:
            return self.info[issue_name]
        else:
            raise ValueError(
                f"issue_name {issue_name} not found in self.info. These have not been computed yet."
            )

    def find_issues(self, issue_types: list = None, verbose=True, **kwargs):
        """
        Audits self.image_files
        For issue checks performed on each image (i.e. brightness, odd size, potential occlusion)
            for each image, compute the score for each check
        For issue checks depending on entire image dataset (i.e. duplicates)
            maintain data structures storing info on the entire image dataset
            for each image, take these data structures as input and update accordingly
        calls analyze_scores to perform analysis and obtain data for output


        Parameters
        ----------
        verbose: bool, Default = True
        A boolean variable where iff True, show a subset of images (<= num_preview) with issues.

        num_preview: int, Default = 10
        An integer representing the number of images with the issue shown (i.e. Blurry)
        or the number of groups of images shown for issues identified in image groups (i.e. Near Duplicates).
        Set `num_preview` = 0 to not show any image previews.

        threshold: int, Default = 5
        An integer representing the percentile threshold for issue scores below which an 
        image is considered as suffering from that issue.
        A larger threshold values will lead to more images being flagged with issues.

        issue_types: list[str], optional
        A list of strings indicating names of checks bring run.
        Defaults to all issue checks is none specified. 
         

        Returns
        -------
        (issue_dict, issue_df): tuple

        issue_dict: dict[str, list]
        a dictionary where keys are string names of issue checks
        and respective values are a list of images indices suffering from the given issue ordered by severity (high to low)

        issue_df: pd.DataFrame
        a pandas dataframe where each row represents a image index
        each column represents a property of the image
        For binary checks (i.e. duplicated images), each cell contains a boolean of 1 represents if an image suffer from the issue, 0 otherwise
        For other checks, each cell contains a score between 0 and 1 (with low score being severe issue)

        misc_info: dict[str, Any]
        a dictionary where keys are string names of miscellaneous info and values are the info stored in the most intuitive data structure.
        """
        issue_kwargs = {
            "todo_fake_arg": 777,
        }

        self.verbose = verbose

        num_preview = 10 # TODO: remove num_preview from this method

        if num_preview <= 0:
            verbose = False


        # Issues to be detected
        if issue_types is None:  # defaults to run all checks
            all_issues = list(POSSIBLE_ISSUES.keys())
            self.issue_types = dict(zip(all_issues, [True] * len(all_issues))) #todo: check why we need this
        else:
            for c in issue_types:
                if c not in POSSIBLE_ISSUES:
                    raise ValueError("Not a valid issue check!")
            self.issue_types = dict(zip(issue_types, [True] * len(issue_types)))
        print(f"Checking for {self.issue_types }")
        # Instantiating issue managers
        self.issue_managers = [
            factory(imagelab=self)
            for factory in _IssueManagerFactory.from_list(list(self.issue_types.keys()))
        ]

        """
            issue_score= {
                "Brightness": {
                    "image_0.jpg": 0.4
                    ....
                }
            }
        """
        self.issue_scores = {}
        for issue_type in self.issue_types.keys():
            self.issue_scores[issue_type] = {}

        # self.issue_scores = dict(zip(self.issue_types.keys(), [{}] * len(self.issue_types.keys())))  # dict where keys are string names of issues, values are list in image order of scores between 0 and 1
        # print('ISSUE SCORES!', self.issue_scores)
        self.results = pd.DataFrame(self.image_files, columns=['image_name'])

        # populates self.issue_scores{} and self.issue_info{}
        count=0
        for image_name in tqdm(self.image_files):
            img = Image.open(os.path.join(self.path, image_name))
            # img.thumbnail(self.thumbnail_size)
            for issue_manager in self.issue_managers:
                issue_manager.find_issues(img, image_name, **issue_kwargs)
            count+=1





    def aggregate(self, thresholds):
        self.thresholds = thresholds
        if len(self.issue_scores) == 0 or self.results is None:
            print('Call find_issues() first.')
            return

        for issue_manager in self.issue_managers:
            issue_manager.aggregate()

    def summary(self, num_preview=10):
        if self.results is None or self.results.shape[0] == 1:
            print('Call find_issues() then aggregate() to get summary')
            return

        if num_preview > 0:
            for issue_manager in self.issue_managers:
                issue_manager.visualize(num_preview)

        return self.results

    def get_overall_scores(self):
        print('TODO: get_overall_scores()')
        return
        # overall_scores = (
        #     []
        # )  # compute overall score with element-wise multiplication of all nonbinary scores
        # for c1 in self.issue_types.keys():
        #     if c1 not in DATASET_WIDE_ISSUES:
        #         if overall_scores == []:
        #             overall_scores = np.array(self.issue_scores[c1])
        #         else:
        #             overall_scores *= np.array(self.issue_scores[c1])
        # issue_data["Overall Score"] = list(overall_scores)
        # issue_df = pd.DataFrame(issue_data)
        # self.issue_df = issue_df
        #
        # return (self.issue_info, issue_df)

class IssueManager(ABC):
    """Base class for managing issues in Imagelab."""

    def __init__(self, imagelab: Imagelab):
        self.imagelab = imagelab

    def __repr__(self):
        class_name = self.__class__.__name__
        return class_name

    def __str__(self):
        class_name = self.__class__.__name__
        return class_name

    @abstractmethod
    def find_issues(self, /, *args, **kwargs):
        """Finds issues in this Lab."""
        raise NotImplementedError

    @abstractmethod
    def update_info(self, /, *args, **kwargs) -> None:
        """Updates the info attribute of this Lab."""
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, /, *args, **kwargs) -> None:
        """Aggregates total scores the info attribute of this Lab."""
        raise NotImplementedError

    @abstractmethod
    def visualize(self, /, *args, **kwargs) -> None:
        raise NotImplementedError

# THIS IS A DATASET WIDE ISSUE TEMPLATE
# testing for check_duplicated
class DatasetWideIssueManager(IssueManager):

    # TODO: Add `results_key = "label"` to this class
    # TODO: Add `info_keys = ["label"]` to this class
    def __init__(self, imagelab: Imagelab):
        super().__init__(imagelab)
        self.issue_name = 'Duplicated'

    def find_issues(self, img, image_name, count, **kwargs) -> float:
        issue_info, misc_info = check_duplicated(img, image_name, count, self.imagelab.issue_info, self.imagelab.misc_info)
        self.update_info(image_name, issue_info, misc_info)
        return self.imagelab.issue_info, self.imagelab.misc_info

    def update_info(self, image_name, issue_info, misc_info, **kwargs) -> None:
        self.imagelab.issue_info = issue_info
        self.imagelab.misc_info = misc_info

    def aggregate(self):
        scores = self.imagelab.issue_scores[self.issue_name]
        if self.imagelab.verbose:
            print(f"Issue {self.issue_name} has {len(self.imagelab.issue_info[self.issue_name])} issues")
        return scores

    def visualize(self, num_preview):
        image_ids = display_images(self.imagelab.issue_info[self.issue_name], num_preview)
        for x in image_ids:  # show the first num_preview duplicate images (if exists)
            try:
                img = Image.open(
                    os.path.join(self.imagelab.path, self.imagelab.image_files[x])
                )
                img.show()
            except:
                break

# THIS IS A DATASET WIDE ISSUE
# testing for check_duplicated
class CheckNearDuplicatesIssueManager(IssueManager):

    # TODO: Add `results_key = "label"` to this class
    # TODO: Add `info_keys = ["label"]` to this class
    def __init__(self, imagelab: Imagelab):
        super().__init__(imagelab)
        self.issue_name = 'Near Duplicates'

    def find_issues(self, img, image_name, count, **kwargs) -> float:
        issue_info, misc_info = check_near_duplicates(img, image_name, count, self.imagelab.issue_info, self.imagelab.misc_info)
        self.update_info(image_name, issue_info, misc_info)
        return self.imagelab.issue_info, self.imagelab.misc_info

    def update_info(self, image_name, issue_info, misc_info, **kwargs) -> None:
        self.imagelab.issue_info = issue_info
        self.imagelab.misc_info = misc_info

    def aggregate(self):
        scores = self.imagelab.issue_scores[self.issue_name]
        if self.imagelab.verbose:
            print(f"Issue {self.issue_name} has {len(self.imagelab.issue_info[self.issue_name])} issues")
        return scores

    def visualize(self, num_preview):
        image_ids = display_images(self.imagelab.issue_info[self.issue_name],num_preview)
        for x in image_ids:  # show the first num_preview duplicate images (if exists)
            try:
                img = Image.open(
                    os.path.join(self.imagelab.path, self.imagelab.image_files[x])
                )
                img.show()
            except:
                break


# THIS IS NOT A DATASET WIDE ISSUE
# testing for check_odd_size
class DatasetSkinnyIssueManager(IssueManager):

    # TODO: Add `results_key = "label"` to this class
    # TODO: Add `info_keys = ["label"]` to this class
    def __init__(self, imagelab: Imagelab):
        super().__init__(imagelab)
        self.issue_name = 'Odd Size'

    def find_issues(self, img, image_name, count, **kwargs) -> pd.DataFrame:
        score = check_odd_size(img)
        self.update_info(image_name, score)
        return score

    def update_info(self, image_name, score, **kwargs) -> None:
        self.imagelab.issue_scores[self.issue_name][image_name] = score

    def aggregate(self):
        scores = self.imagelab.issue_scores[self.issue_name]

        self.imagelab.results[f'{self.issue_name} raw_score'] = self.imagelab.results['image_name'].map(scores)
        self.imagelab.results[f'{self.issue_name} zscore'] = get_zscores(self.imagelab.results[f'{self.issue_name} raw_score'].tolist())
        self.imagelab.results[f'{self.issue_name} bool'] = get_is_issue(self.imagelab.results[f'{self.issue_name} zscore'].tolist(), self.imagelab.thresholds)

        if self.imagelab.verbose:
            print(f"Issue {self.issue_name} has {np.sum(self.imagelab.results[f'{self.issue_name} bool'].tolist())} issues")

    def visualize(self, num_preview=10):
        results_col = self.imagelab.results[f'{self.issue_name} bool']
        issue_indices = self.imagelab.results.index[results_col].tolist()

        for ind in display_images(issue_indices, num_preview):  # show the top 10 issue images (if exists)
            try:
                img = Image.open(os.path.join(self.imagelab.path, self.imagelab.image_files[ind]))
                img.show()
            except:
                break

class EntropyIssueManager(IssueManager):

    # TODO: Add `results_key = "label"` to this class
    # TODO: Add `info_keys = ["label"]` to this class
    def __init__(self, imagelab: Imagelab):
        super().__init__(imagelab)
        self.issue_name = 'Entropy'

    def find_issues(self, img, image_name, **kwargs) -> pd.DataFrame:
        score = check_entropy(img)
        self.update_info(image_name, score)
        return score

    def update_info(self, image_name, score, **kwargs) -> None:
        print(f'Update info called for {image_name} {self.issue_name}')
        self.imagelab.issue_scores[self.issue_name][image_name] = score

    def aggregate(self):
        scores = self.imagelab.issue_scores[self.issue_name]

        self.imagelab.results[f'{self.issue_name} raw_score'] = self.imagelab.results['image_name'].map(scores)
        self.imagelab.results[f'{self.issue_name} zscore'] = get_zscores(self.imagelab.results[f'{self.issue_name} raw_score'].tolist())
        self.imagelab.results[f'{self.issue_name} bool'] = get_is_issue(self.imagelab.results[f'{self.issue_name} zscore'].tolist(), self.imagelab.thresholds)
class BrightnessIssueManager(IssueManager):

    # TODO: Add `results_key = "label"` to this class
    # TODO: Add `info_keys = ["label"]` to this class
    # todo: remove imagelab as a parameter to make it lightweight
    def __init__(self, imagelab: Imagelab):
        super().__init__(imagelab)
        self.issue_name = 'Brightness'

    def find_issues(self, img, image_name, count, **kwargs) -> pd.DataFrame:
        score = check_brightness(img)
        self.update_info(image_name, score)
        return score

    def update_info(self, image_name, score, **kwargs) -> None:
        print(f'Update info called for {image_name} {self.issue_name}')
        self.imagelab.issue_scores[self.issue_name][image_name] = score

    def aggregate(self):
        scores = self.imagelab.issue_scores[self.issue_name]

        self.imagelab.results[f'{self.issue_name} raw_score'] = self.imagelab.results['image_name'].map(scores)
        self.imagelab.results[f'{self.issue_name} zscore'] = get_zscores(self.imagelab.results[f'{self.issue_name} raw_score'].tolist())
        self.imagelab.results[f'{self.issue_name} bool'] = get_is_issue(self.imagelab.results[f'{self.issue_name} zscore'].tolist(), self.imagelab.thresholds)

class BlurredIssueManager(IssueManager):

    # TODO: Add `results_key = "label"` to this class
    # TODO: Add `info_keys = ["label"]` to this class
    # todo: remove imagelab as a parameter to make it lightweight
    def __init__(self, imagelab: Imagelab):
        super().__init__(imagelab)
        self.issue_name = 'Blurred'

    def find_issues(self, img, image_name, count, **kwargs) -> pd.DataFrame:
        score = check_blurriness(img)
        self.update_info(image_name, score)
        return score

    def update_info(self, image_name, score, **kwargs) -> None:
        print(f'Update info called for {image_name} {self.issue_name}')
        self.imagelab.issue_scores[self.issue_name][image_name] = score

    def aggregate(self):
        scores = self.imagelab.issue_scores[self.issue_name]

        self.imagelab.results[f'{self.issue_name} raw_score'] = self.imagelab.results['image_name'].map(scores)
        self.imagelab.results[f'{self.issue_name} zscore'] = get_zscores(self.imagelab.results[f'{self.issue_name} raw_score'].tolist())
        self.imagelab.results[f'{self.issue_name} bool'] = get_is_issue(self.imagelab.results[f'{self.issue_name} zscore'].tolist(), self.imagelab.thresholds)

# Construct concrete issue manager with a from_str method
class _IssueManagerFactory:
    """Factory class for constructing concrete issue managers."""
    #todo: convert these strings to constants
    types = {
        "Duplicated": DatasetWideIssueManager,
        "Odd Size": DatasetSkinnyIssueManager,
        "Brightness": BrightnessIssueManager,
        "Blurred": BlurredIssueManager,
        "Potential Occlusion": DatasetSkinnyIssueManager,
        "Entropy": EntropyIssueManager,
        "Near Duplicates": CheckNearDuplicatesIssueManager,
    }

    @classmethod
    def from_str(cls, issue_type: str) -> Type[IssueManager]:
        """Constructs a concrete issue manager from a string."""
        if isinstance(issue_type, list):
            raise ValueError(
                "issue_type must be a string, not a list. Try using from_list instead."
            )
        if issue_type not in cls.types:
            raise ValueError(f"Invalid issue type: {issue_type}")
        return cls.types[issue_type]

    @classmethod
    def from_list(cls, issue_types: List[str]) -> List[Type[IssueManager]]:
        """Constructs a list of concrete issue managers from a list of strings."""
        return [cls.from_str(issue_type) for issue_type in issue_types]