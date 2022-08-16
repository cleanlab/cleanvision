import os, warnings
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
from image_data_quality.utils.utils import analyze_scores, get_sorted_images, display_images, get_total_num_issues

POSSIBLE_ISSUES = {
    "Brightness": check_brightness,
    "Odd Size": check_odd_size,
    "Potential Occlusion": check_entropy,
    "Potential Static": check_static,
    "Blurry": check_blurriness,
    "Duplicated": check_duplicated,
    "Near Duplicates": check_near_duplicates,
}
DATASET_WIDE_ISSUES = {
    "Duplicated",
    "Near Duplicates",
}  # issues requiring info. from entire dataset

MISC_INFO = {
    "Brightness": ['Brightness sorted z-scores'],
    "Odd Size": ['Odd Size sorted z-scores'],
    "Potential Occlusion": ['Potential Occlusion sorted z_scores'],
    "Potential Static": ['Potential Static sorted z_scores'],
    "Blurry": ['Blurry sorted z-scores'],
    "Duplicated": ['Image Hashes', 'Hash to Image', 'Duplicate Image Groups'],
    "Near Duplicates": ['Near Duplicate Imagehashes', 'Imagehash to Image', 'Near Duplicate Image Groups']
}

class ImageDataset:
    """
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

    issues_checked: dict, optional
        A dictionary where keys are string names of checks being run, and respective values
        are another dictionary containing hyperparameters. 
        Values are empty dict by default
         
    """
    num_images = 0
    issue_summary = {}
    """key: issue name string, value: list of indices of images with this issue (dict)"""
    misc_info = {}
    """key: miscellanous info name string, value: intuitive data structure containing that info (dict)"""
    total_num_issues = 0 #TODO: integer representing total number of issue images
    def __init__(
        self, path: str = None, image_files: list=None, thumbnail_size: tuple=None, issues_checked: dict = None
    ):
        if path is None:
            self.path = os.getcwd()
        else:
            self.path = path
        if image_files is None:
            self.image_files = get_sorted_images(self.path)
            self.num_images = len(self.image_files)
        else:
            self.image_files = sorted(image_files)
            self.num_images = len(self.image_files)
        if thumbnail_size is None:
            self.thumbnail_size = (128, 128)
        else: 
            self.thumbnail_size = thumbnail_size
        #if issues_checked is None:
            #self.issues_checked = {}
        #else:
            #self.issues_checked = issues_checked
        # TODO: revisit default value for thumbnail_size
    
        
    def __repr__(self):
        return "ImageDataset(num_images = " + str(self.num_images) + ", path = " + str(self.path) + ", num_images_with_issue = " + str(self.total_num_issues)+")"
  
    def __str__(self):
        return "num_images = " + str(self.num_images) + ", path = " + str(self.path) + ", num_images_with_issue = " + str(self.total_num_issues)
       
    
    def find_issues(self, threshold = None, issues_checked: dict = None, verbose=True, num_preview = None, **kwargs):
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
        or the number of groups of images shown for issues identified in image groups (i.e. Near Duplicates)

        threshold: int, Default = 5
        An integer representing the percentile threshold for issue scores below which an 
        image is considered as suffering from that issue.
        A larger threshold values will lead to more images being flagged with issues.

        issues_checked: dict, optional
        A dictionary where keys are string names of checks being run, and respective values
        are another dictionary containing hyperparameters. 
        Values are empty dict by default
         

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
        if threshold is None:
            threshold = 5
        if num_preview <= 0:
            verbose = False
        elif num_preview is None:
            num_preview = 10
        if issues_checked is None:  # defaults to run all checks
            self.issues_checked = {k:{} for k in POSSIBLE_ISSUES.keys()}
        else:
            for c in issues_checked:
                if c not in POSSIBLE_ISSUES:
                    raise ValueError("Not a valid issue check!")
            self.issues_checked = {**self.issues_checked, **issues_checked}
        count = 0
        issue_scores = (
            {}
        )  # dict where keys are string names of issues, values are list in image order of scores between 0 and 1
        for image_name in tqdm(self.image_files):
            img = Image.open(os.path.join(self.path, image_name))
            img.thumbnail(self.thumbnail_size)
            for c in self.issues_checked:  # run each check for each image
                try:
                    c_kwargs = self.issues_checked[c]
                    if c in DATASET_WIDE_ISSUES:
                        (self.issue_summary, self.misc_info) = POSSIBLE_ISSUES[c](
                                img, image_name, count, self.issue_summary, self.misc_info, **c_kwargs
                        )
                    else:
                        issue_scores.setdefault(c, []).append(POSSIBLE_ISSUES[c](img, **c_kwargs))
                except Exception as e:
                    warnings.warn("issue_check_func threw exception, this check was not run on your data."
 				    "Exception: " + e
    		        )
                    del self.issues_checked[c]
            count += 1
        if verbose:
            for c in DATASET_WIDE_ISSUES:
                if c in self.issues_checked:
                    if len(self.issue_summary[c]) > 0:
                        print("These images have", c, "issue")
                    else:
                        continue
                    for x in display_images(self.issue_summary[c], num_preview):  # show the first num_preview duplicate images (if exists)
                        try:
                            img = Image.open(
                                os.path.join(self.path, self.image_files[x])
                            )
                            img.show()
                        except:
                            break
        issue_data = {}
        issue_data["Names"] = self.image_files
        for c1 in self.issues_checked:
            if c1 not in DATASET_WIDE_ISSUES:
                analysis = analyze_scores(issue_scores[c1], threshold)
                issue_indices = analysis[0]
                boolean = list(analysis[1].values())
                self.issue_summary[c1] = issue_indices
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
        has_overall = False
        for c1 in self.issues_checked:
            if c1 not in DATASET_WIDE_ISSUES:
                has_overall = True
                if overall_scores == []:
                    overall_scores = np.array(issue_scores[c1])
                else:
                    overall_scores *= np.array(issue_scores[c1])
        if has_overall:
            issue_data["Overall Score"] = list(overall_scores)
        issue_df = pd.DataFrame(issue_data)
        issue_names = self.issue_summary.keys()
        issue_indices = list(self.issue_summary.values())
        self.total_num_issues = get_total_num_issues(self.issue_summary)
        num_examples = []
        info_list = [MISC_INFO[check] for check in issue_names]
        for e in issue_indices: 
            if type(e[0]) is list: #if nested list
                num_examples.append(len([item for l in e for item in l]))
            else:
                num_examples.append(len(e))
        #compile information and convert self.issue_summary into pandas Dataframe
        self.issue_summary = pd.DataFrame({"Issue Names": issue_names, "Number with Issue": num_examples, "Issue Indices":issue_indices, "Misc. Info Content": info_list})
        if verbose:
            print(self.issue_summary)
        return issue_df
