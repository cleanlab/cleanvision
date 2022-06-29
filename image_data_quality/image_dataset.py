import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from image_data_quality.issue_checks import (
    check_brightness,
    check_odd_size,
    check_entropy,
    check_duplicated,
    check_near_duplicates,
)
from image_data_quality.utils.utils import analyze_scores, get_sorted_images

possible_issues = {
    "Duplicated": check_duplicated,
    "Brightness": check_brightness,
    "Odd size": check_odd_size,
    "Potential occlusion": check_entropy,
    "Near duplicates": check_near_duplicates,
}
dataset_wide_issues = {
    "Duplicated",
    "Near duplicates",
}  # issues requiring info. from entire dataset


def flattenList(nestedList):
    """
    Recursively returns a flattened list of a nested list.
    """

    # check if list is empty
    if not (bool(nestedList)):
        return nestedList

    # to check instance of list is empty or not
    if isinstance(nestedList[0], list):

        # call function with sublist as argument
        return flattenList(*nestedList[:1]) + flattenList(nestedList[1:])

    # call function with sublist as argument
    return nestedList[:1] + flattenList(nestedList[1:])


class ImageDataset:
    def __init__(
        self, path = None, image_files=None, thumbnail_size=None, issues_checked=None
    ):
        if path is None:
            self.path = os.getcwd()
        else:
            self.path = path
        if image_files is None:
            self.image_files = get_sorted_images(self.path)
        else:
            self.image_files = image_files
        if thumbnail_size is None:
            self.thumbnail_size = (128, 128)
        # TODO: revisit default value for thumbnail_size
        if issues_checked is None:  # defaults to run all checks
            self.issues_checked = list(possible_issues.keys())
        else:
            for c in issues_checked:
                if c not in possible_issues:
                    raise ValueError("Not a valid issue check!")
            self.issues_checked = issues_checked
        self.issue_info = (
            {}
        )  # key: issue name string, value: list of indices of images with this issue
        self.misc_info = (
            {}
        )  # key: misc info name string, value: intuitive data structure containing that info

    def audit_images(self, verbose=True, num_preview=10):
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
        count = 0
        issue_scores = (
            {}
        )  # dict where keys are string names of issues, values are list in image order of scores between 0 and 1
        for image_name in tqdm(self.image_files):
            img = Image.open(os.path.join(self.path, image_name))
            img.thumbnail(self.thumbnail_size)
            for c in self.issues_checked:  # run each check for each image
                if c in dataset_wide_issues:
                    (self.issue_info, self.misc_info) = possible_issues[c](
                        img, image_name, count, self.issue_info, self.misc_info
                    )
                else:
                    issue_scores.setdefault(c, []).append(possible_issues[c](img))
            count += 1
        if verbose:
            for c in dataset_wide_issues:
                if len(self.issue_info[c]) > 0:
                    print("These images have", c, "issue")
                else:
                    continue
                for x in range(
                    num_preview
                ):  # show the first 10 duplicate images (if exists)
                    img = Image.open(
                        os.path.join(self.path, self.image_files[flattenList(self.issue_info[c])[x]])
                    )
                    try:
                        img = Image.open(
                            os.path.join(self.path, self.image_files[flattenList(self.issue_info[c])[x]])
                        )
                        img.show()
                    except:
                        break
        issue_data = {}
        issue_data["Names"] = self.image_files
        for c1 in self.issues_checked:
            if c1 not in dataset_wide_issues:
                analysis = analyze_scores(issue_scores[c1])
                issue_indices = analysis[0]
                boolean = list(analysis[1].values())
                self.issue_info[c1] = issue_indices
                issue_data[c1 + " issue"] = boolean
                issue_data[c1 + " score"] = issue_scores[c1]
                if verbose:
                    if len(issue_indices) > 0:
                        print("These images have", c1, "issue")
                        for ind in range(
                            num_preview
                        ):  # show the top 10 issue images (if exists)
                            try:
                                img = Image.open(os.path.join(self.path, self.image_files[issue_indices[ind]]))
                                img.show()
                            except:
                                break
        overall_scores = (
            []
        )  # compute overall score with element-wise multiplication of all nonbinary scores
        for c1 in self.issues_checked:
            if c1 not in dataset_wide_issues:
                if overall_scores == []:
                    overall_scores = np.array(issue_scores[c1])
                else:
                    overall_scores *= np.array(issue_scores[c1])
        issue_data["Overall Score"] = list(overall_scores)
        issue_df = pd.DataFrame(issue_data)
        return (self.issue_info, issue_df)
