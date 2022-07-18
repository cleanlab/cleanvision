import os
import numpy as np
import pandas as pd
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
from image_data_quality.utils.utils import analyze_scores, get_sorted_images, display_images

POSSIBLE_ISSUES = {
    "Duplicated": check_duplicated,
    "Brightness": check_brightness,
    "Odd Size": check_odd_size,
    "Blurry": check_blurriness,
    "Potential Occlusion": check_entropy,
    "Potential Static": check_static,
    "Near Duplicates": check_near_duplicates,
}
DATASET_WIDE_ISSUES = {
    "Duplicated",
    "Near Duplicates",
}  # issues requiring info. from entire dataset

class ImageDataset:
    """
    Auto-detects issues in image data that may cause problems in training. 
    This class takes in a dataset of images and performs targeted checks 
    to report the indices of problematic images and other relevant information. 

    Parameters
    ----------
    path: string
      path to folder where image dataset is located
      If not provided with path, default set to current working directory.

    image_files: list of strings
      a list of filenames in the image dataset sorted numerically and alphabetically
      If provided with filenames, sorts using built-in sorting function.
      Default set to list of all images in the dataset.
    
    thumbnail_size: tuple of 2 integers
      a tuple specifying the size of the thumbnail in which image analysis is based on. 
      If not provided with the 2-tuple, default set to (128, 128). 

    self.issue_info: a dictionary
      a dictionary where keys are strings describing issue names, 
      and respective values are either a list of indices of images with this issue, 
      or a nested list containing groups of images with the same issue. 
        
    self.misc_info: a dictionary
      a dictionary where keys are strings describing names of miscellaneous information, 
    """
    def __init__(
        self, path: str=None, image_files: list=None, thumbnail_size: tuple=None
    ):
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
        # TODO: revisit default value for thumbnail_size
        self.issue_info = (
            {}
        )  # key: issue name string, value: list of indices of images with this issue
        self.misc_info = (
            {}
        )  # key: misc info name string, value: intuitive data structure containing that info

    def find_issues(self, verbose=True, num_preview=10, threshold = 5, issues_checked: list = None, **kwargs):
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
        verbose: bool (defaults to True)
        a boolean variable where iff True, show a subset of images (<= 10) with issues

        num_preview: int
        an integer representing the number of images with the issue shown 
        or the number of issue image groups shown

        Returns
        -------
        a tuple: (issue_dict, issue_df)

        issue_dict: dict
        a dictionary where keys are string names of issue checks
        and respective values are a list of images indices suffering from the given issue ordered by severity (high to low)

        issue_df: pd.DataFrame
        a pandas dataframe where each row represents a image index
        each column represents a property of the image
        For binary checks (i.e. duplicated images), each cell contains a boolean of 1 represents if an image suffer from the issue, 0 otherwise
        For other checks, each cell contains a score between 0 and 1 (with low score being severe issue)

        misc_info: dict
        a dictionary where keys are string names of miscellaneous info and values are the info stored in the most intuitive data structure.
        """
        if issues_checked is None:  # defaults to run all checks
            issues_checked = list(POSSIBLE_ISSUES.keys())
        else:
            for c in issues_checked:
                if c not in POSSIBLE_ISSUES:
                    raise ValueError("Not a valid issue check!")
        count = 0
        issue_scores = (
            {}
        )  # dict where keys are string names of issues, values are list in image order of scores between 0 and 1
        for image_name in tqdm(self.image_files):
            img = Image.open(os.path.join(self.path, image_name))
            img.thumbnail(self.thumbnail_size)
            for c in issues_checked:  # run each check for each image
                if c in DATASET_WIDE_ISSUES:
                    if c in kwargs:
                        (self.issue_info, self.misc_info) = POSSIBLE_ISSUES[c](
                            img, image_name, count, self.issue_info, self.misc_info, **kwargs[c]
                        )
                else:
                    issue_scores.setdefault(c, []).append(POSSIBLE_ISSUES[c](img))
            count += 1
        if verbose:
            for c in DATASET_WIDE_ISSUES:
                if c in issues_checked:
                    if len(self.issue_info[c]) > 0:
                        print("These images have", c, "issue")
                    else:
                        continue
                    for x in display_images(self.issue_info[c], num_preview):  # show the first num_preview duplicate images (if exists)
                        try:
                            img = Image.open(
                                os.path.join(self.path, self.image_files[x])
                            )
                            img.show()
                        except:
                            break
        issue_data = {}
        issue_data["Names"] = self.image_files
        for c1 in issues_checked:
            if c1 not in DATASET_WIDE_ISSUES:
                analysis = analyze_scores(issue_scores[c1], threshold)
                issue_indices = analysis[0]
                boolean = list(analysis[1].values())
                self.issue_info[c1] = issue_indices
                issue_data[c1 + " issue"] = boolean
                issue_data[c1 + " score"] = issue_scores[c1]
                self.misc_info[c1 + " sorted z-scores"] = analysis[2]
                if verbose:
                    if len(issue_indices) > 0:
                        print("These images have", c1, "issue")
                        for ind in display_images(issue_indices, num_preview):  # show the top 10 issue images (if exists)
                            try:
                                img = Image.open(os.path.join(self.path, self.image_files[ind]))
                                img.show()
                            except:
                                break
        overall_scores = (
            []
        )  # compute overall score with element-wise multiplication of all nonbinary scores
        for c1 in issues_checked:
            if c1 not in DATASET_WIDE_ISSUES:
                if overall_scores == []:
                    overall_scores = np.array(issue_scores[c1])
                else:
                    overall_scores *= np.array(issue_scores[c1])
        issue_data["Overall Score"] = list(overall_scores)
        issue_df = pd.DataFrame(issue_data)
        return (self.issue_info, issue_df)
