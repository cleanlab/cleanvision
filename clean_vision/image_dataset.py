import os
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from queue import Queue
from typing import List, Type

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from clean_vision.issue_checks import (
    check_odd_size,
    get_image_hash,
    get_near_duplicate_hash,
    get_brightness_score,
    check_entropy,
    check_blurriness,
    check_grayscale,
    find_hot_pixels,
)
from clean_vision.utils.utils import (
    get_sorted_images,
    display_images,
    get_is_issue,
)

POSSIBLE_ISSUES = set(
    [
        "Duplicated",
        "DarkImages",
        "LightImages",
        "AspectRatio",
        "Blurred",
        "Entropy",
        "NearDuplicates",
        "Grayscale",
        # "HotPixels",
    ]
)

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
        self, path: str = None, image_files: list = None, thumbnail_size: tuple = None
    ) -> None:
        if path is None:
            self.path = os.getcwd()
        else:
            self.path = path
        if image_files is None:
            self.image_files = get_sorted_images(self.path)
            self.image_indices = {
                self.image_files[i]: i for i in range(len(self.image_files))
            }
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
        self.thresholds = {
            "Duplicated": 0.5,
            "DarkImages": 0.22,
            "LightImages": 0.05,
            "AspectRatio": 0.5,
            "Blurred": 0.3,
            "Entropy": 0.3,
            "NearDuplicates": 0.5,
            "Grayscale": 0.5,
            "HotPixels": 0.5,
        }
        self.hash_image_map = {}
        self.near_hash_image_map = {}
        self.color_channels = {}

    def __repr__(self):
        """What is displayed in console if user executes: >>> imagelab"""
        if self.issue_info == {}:
            num_issues = None
        else:
            num_issues = 0
            for check in self.issue_info.values():
                if type(check[0]) == list:
                    flat_issue = []
                    for lst in check:
                        flat_issue += lst
                    num_issues += len(flat_issue)
                else:
                    num_issues += len(check)
        display_str = (
            "ImageDataset(num_images = "
            + str(len(self.image_files))
            + ", path = "
            + str(self.path)
            + ", num_images_with_issue = "
            + str(num_issues)
            + ")"
        )
        # Useful info could be: num_images, path, number of images with issues identified so far (numeric or None if issue-finding not run yet).
        return display_str

    def __str__(self):
        """What is displayed if user executes: print(imagelab)"""
        return self.__repr__()[
            13:-1
        ]  # display_info could be same information as above in display_str without the ImageDataset(...) wrapper text.

    # DEPRECATE FOR NOW
    # def get_info(self, issue_name) -> Any:
    #     """Returns dict of info about a specific issue, or None if this issue does not exist in self.info.
    #     Internally fetched from self.info[issue_name] and prettified.
    #     Keys might include: number of examples suffering from issue, indicates of top-K examples most severely suffering,
    #     other misc stuff like which sets of examples are duplicates if the issue=="duplicated".
    #     """
    #     if issue_name in self.info:
    #         return self.info[issue_name]
    #     else:
    #         raise ValueError(
    #             f"issue_name {issue_name} not found in self.info. These have not been computed yet."
    #         )

    def evaluate(self, issue_types: list = None, verbose=True, **kwargs):
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

        self.verbose = verbose  # TODO: fix verbose

        # Issues to be detected
        if issue_types is None:  # defaults to run all checks
            all_issues = list(POSSIBLE_ISSUES)
            self.issue_types = dict(
                zip(all_issues, [True] * len(all_issues))
            )  # todo: check why we need this
        else:
            for c in issue_types:
                if c not in POSSIBLE_ISSUES:
                    raise ValueError("Not a valid issue check!")
            self.issue_types = dict(zip(issue_types, [True] * len(issue_types)))
        print(f"Checking for {', '.join(self.issue_types.keys())}")

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
            self.issue_scores[issue_type] = OrderedDict(
                [(image_file, None) for image_file in self.image_files]
            )

        self.results = pd.DataFrame(self.image_files, columns=["image_name"])

        num_cpus = os.cpu_count()
        loaded_images = Queue(maxsize=num_cpus)
        worker_images = self.image_files.copy()

        def worker():
            while worker_images:
                name = worker_images.pop()
                im = Image.open(os.path.join(name))
                loaded_images.put((im, name))

        for _ in range(num_cpus):
            t = threading.Thread(target=worker, daemon=True)
            t.start()

        for i in tqdm(range(len(self.image_files))):
            img, image_name = loaded_images.get()
            if img.mode is not None:
                self.color_channels[img.mode] = self.color_channels.get(img.mode, 0) + 1
            for issue_manager in self.issue_managers:
                issue_manager.find_issues(img, image_name, **issue_kwargs)

    def aggregate(self, thresholds=None):
        if thresholds is not None:
            for issue_name, t in thresholds.items():
                self.thresholds[issue_name] = t
        if len(self.issue_scores) == 0 or self.results is None:
            print("Call find_issues() first.")
            return

        for issue_manager in self.issue_managers:
            issue_manager.aggregate(self.thresholds[issue_manager.issue_name])

    def get_thresholds(self):
        return self.thresholds

    def summary(self):
        if self.results is None or self.results.shape[0] == 1:
            print("Call find_issues() then aggregate() to get summary().")
            return

        bool_columns = [col for col in self.results if col.endswith("bool")]
        bool_df = self.results[bool_columns].copy()
        col_rename = [col[:-5] for col in bool_columns]
        bool_df = bool_df.rename(
            columns=dict(zip(bool_columns, col_rename))
        )  # remove "bool" out of column names

        score_columns = [col for col in self.results if col.endswith("score")]
        score_df = self.results[score_columns].copy()
        col_rename = [col[:-6] for col in score_columns]
        score_df = score_df.rename(
            columns=dict(zip(score_columns, col_rename))
        )  # remove "zscore" out of column names

        # issue_score_sum = len(self.image_files) - score_df.sum()
        # total_sum = issue_score_sum.sum()
        # score_df = (score_df.shape[0] - score_df).sum() / score_df.mean()

        summary_results = pd.DataFrame(
            {
                "Issues": bool_df.sum(),
                "Percent of Data": round(
                    bool_df.sum() / self.results.shape[0] * 100, 2
                ),
                "Issue Intensity": bool_df.sum() / bool_df.sum().sum(),
            }
        )

        summary_results = summary_results.sort_values(
            by=["Issue Intensity"], ascending=False
        )
        print(
            f"Color spaces in the  dataset\n========================\n{self.color_channels}\n"
        )
        print(f"Issue Summary\n========================\n{summary_results}\n")
        return summary_results, self.results

    def visualize(self, num_preview=10, verbose=True):
        if self.results is None or self.results.shape[0] == 1:
            print("Call find_issues() then aggregate() before visualize().")
            return

        # TODO: num issues can be a variable in each
        if num_preview is None or num_preview > 0:
            for issue_manager in self.issue_managers:
                # if verbose:
                #     print(f"Found {issue_manager.num_issues} images with the issue {issue_manager.issue_name}")
                if issue_manager.num_issues > 0:
                    issue_manager.visualize(num_preview)

        return self.results

    def get_overall_scores(self):
        # print('TODO: get_overall_scores()')
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
        self.num_issues = None

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

    # @abstractmethod
    def mark_bool_issues(self, scores, threshold):
        return scores < threshold

    @abstractmethod
    def aggregate(self, threshold, /, *args, **kwargs) -> None:
        """Aggregates total scores the info attribute of this Lab."""
        raise NotImplementedError

    @abstractmethod
    def visualize(self, /, *args, **kwargs) -> None:
        raise NotImplementedError


# THIS IS A DATASET WIDE ISSUE TEMPLATE
# testing for check_duplicated
class DuplicatedIssueManager(IssueManager):

    # TODO: Add `results_key = "label"` to this class
    # TODO: Add `info_keys = ["label"]` to this class
    def __init__(self, imagelab: Imagelab):
        super().__init__(imagelab)
        self.issue_name = "Duplicated"

    def find_issues(self, img, image_name, **kwargs) -> float:
        img_hash = get_image_hash(img)
        self.update_info(image_name, img_hash)

    def update_info(self, image_name, img_hash, **kwargs) -> None:
        if img_hash in self.imagelab.hash_image_map:
            self.imagelab.hash_image_map[img_hash].append(image_name)
        else:
            self.imagelab.hash_image_map[img_hash] = [image_name]

    def aggregate(self, threshold):
        duplicated_images = set()
        for hash, img_list in self.imagelab.hash_image_map.items():
            if len(img_list) > 1:
                duplicated_images.update(img_list)
        for img_name in self.imagelab.issue_scores[self.issue_name].keys():
            self.imagelab.issue_scores[self.issue_name][img_name] = (
                0 if img_name in duplicated_images else 1
            )

        raw_scores = np.array(
            list(self.imagelab.issue_scores[self.issue_name].values())
        )
        self.imagelab.results[
            f"{self.issue_name} score"
        ] = raw_scores  # 0: is duplicated 1 is not
        self.imagelab.results[f"{self.issue_name} bool"] = self.mark_bool_issues(
            raw_scores, threshold
        )
        self.num_issues = np.sum(
            self.imagelab.results[f"{self.issue_name} bool"].tolist()
        )

    def visualize(self, num_preview):
        count = 0
        if num_preview is None:
            num_preview = len(self.imagelab.hash_image_map.items())

        for hash, img_list in self.imagelab.hash_image_map.items():
            if len(img_list) > 1:
                for img_name in img_list:
                    ind = self.imagelab.image_indices[img_name]
                    issue_score = self.imagelab.results[
                        self.imagelab.results["image_name"] == img_name
                    ][f"{self.issue_name} score"].tolist()[0]
                    title = f"{self.issue_name} [{issue_score}] {img_name}"
                    img = Image.open(
                        os.path.join(self.imagelab.path, self.imagelab.image_files[ind])
                    )
                    print(title)
                    img.show(title=title)
                count += 1
                if count == num_preview:
                    break

    def get_duplicated_sets(self, n=5):
        duplicated_sets = []
        for hash, img_list in self.imagelab.hash_image_map.items():
            if len(img_list) > 1:
                duplicated_sets.append(img_list)
                if len(duplicated_sets) == n:
                    break
        return duplicated_sets


# THIS IS A DATASET WIDE ISSUE
# testing for check_duplicated
class CheckNearDuplicatesIssueManager(IssueManager):

    # TODO: Add `results_key = "label"` to this class
    # TODO: Add `info_keys = ["label"]` to this class
    def __init__(self, imagelab: Imagelab):
        super().__init__(imagelab)
        self.issue_name = "NearDuplicates"

    def find_issues(self, img, image_name, **kwargs) -> float:
        near_hash = get_near_duplicate_hash(img)
        self.update_info(image_name, near_hash)

    def update_info(self, image_name, near_hash, **kwargs) -> None:
        if near_hash in self.imagelab.near_hash_image_map:
            self.imagelab.near_hash_image_map[near_hash].append(image_name)
        else:
            self.imagelab.near_hash_image_map[near_hash] = [image_name]

    # def mark_bool_issues(self, raw_scores):
    #     return (1 - raw_scores).astype('bool')

    def aggregate(self, threshold):
        duplicated_images = set()
        for hash, img_list in self.imagelab.near_hash_image_map.items():
            if len(img_list) > 1:
                duplicated_images.update(img_list)
        for img_name in self.imagelab.issue_scores[self.issue_name].keys():
            self.imagelab.issue_scores[self.issue_name][img_name] = (
                0 if img_name in duplicated_images else 1
            )

        raw_scores = np.array(
            list(self.imagelab.issue_scores[self.issue_name].values())
        )
        self.imagelab.results[
            f"{self.issue_name} score"
        ] = raw_scores  # 0 is duplicated
        self.imagelab.results[f"{self.issue_name} bool"] = self.mark_bool_issues(
            raw_scores, threshold
        )
        self.num_issues = np.sum(
            self.imagelab.results[f"{self.issue_name} bool"].tolist()
        )

    def visualize(self, num_preview):
        count = 0
        if num_preview is None:
            num_preview = len(self.imagelab.hash_image_map.items())

        for hash, img_list in self.imagelab.near_hash_image_map.items():
            if len(img_list) > 1:
                for img_name in img_list:
                    ind = self.imagelab.image_indices[img_name]
                    issue_score = self.imagelab.results[
                        self.imagelab.results["image_name"] == img_name
                    ][f"{self.issue_name} score"].tolist()[0]
                    title = f"{self.issue_name} [{issue_score}] {img_name}"
                    img = Image.open(
                        os.path.join(self.imagelab.path, self.imagelab.image_files[ind])
                    )
                    img.show(title=title)
                count += 1
                if count == num_preview:
                    break

    def get_duplicated_sets(self, n=5):
        duplicated_sets = []
        for hash, img_list in self.imagelab.near_hash_image_map.items():
            if len(img_list) > 1:
                duplicated_sets.append(img_list)
                if len(duplicated_sets) == n:
                    break
        return duplicated_sets


class EntropyIssueManager(IssueManager):

    # TODO: Add `results_key = "label"` to this class
    # TODO: Add `info_keys = ["label"]` to this class
    def __init__(self, imagelab: Imagelab):
        super().__init__(imagelab)
        self.issue_name = "Entropy"
        self.threshold = 1
        self.t = 0.5

    def find_issues(self, img, image_name, **kwargs) -> pd.DataFrame:
        score = check_entropy(img)
        self.update_info(image_name, score)
        return score

    def update_info(self, image_name, score, **kwargs) -> None:
        self.imagelab.issue_scores[self.issue_name][image_name] = score

    # def mark_bool_issues(self, raw_scores):
    #     return np.array(raw_scores) < 0.3

    def aggregate(self, threshold):
        raw_scores = np.array(
            list(self.imagelab.issue_scores[self.issue_name].values())
        )
        # scores: np.ndarray = 1 - np.exp(-1 * raw_scores * self.t)
        scores: np.ndarray = 0.1 * raw_scores
        scores[scores > 1] = 1
        self.imagelab.results[f"{self.issue_name} score"] = scores
        self.imagelab.results[f"{self.issue_name} bool"] = self.mark_bool_issues(
            scores, threshold
        )
        self.num_issues = np.sum(
            self.imagelab.results[f"{self.issue_name} bool"].tolist()
        )

    def visualize(self, num_preview=10):
        results_col = self.imagelab.results[f"{self.issue_name} bool"]
        scores_col = self.imagelab.results[f"{self.issue_name} score"]
        issue_scores = scores_col.to_numpy()
        issue_indices = np.argsort(issue_scores)
        true_issue_indices = set(self.imagelab.results.index[results_col].tolist())
        issue_indices = [idx for idx in issue_indices if idx in true_issue_indices]

        if (
            num_preview is None
        ):  # TODO: kind of strange I think display_images should be rewritten
            num_preview = len(issue_indices)

        for ind in display_images(
            issue_indices, num_preview
        ):  # show the top 10 issue images (if exists)
            img_name = self.imagelab.image_files[ind]
            issue_score = self.imagelab.results[
                self.imagelab.results["image_name"] == img_name
            ][f"{self.issue_name} score"].tolist()[0]
            title = f"{self.issue_name} [{issue_score}] {img_name}"
            try:
                img = Image.open(
                    os.path.join(self.imagelab.path, self.imagelab.image_files[ind])
                )
                print(title)
                img.show(title=title)
            except:
                break


class DarkImagesIssueManager(IssueManager):

    # TODO: Add `results_key = "label"` to this class
    # TODO: Add `info_keys = ["label"]` to this class
    def __init__(self, imagelab: Imagelab):
        super().__init__(imagelab)
        self.issue_name = "DarkImages"
        self.threshold = 1
        self.t = 1

    def find_issues(self, img, image_name, **kwargs) -> pd.DataFrame:
        score = get_brightness_score(img)
        self.update_info(image_name, score)
        return score

    def update_info(self, image_name, score, **kwargs) -> None:
        self.imagelab.issue_scores[self.issue_name][image_name] = score

    def aggregate(self, threshold):
        raw_scores = np.array(
            list(self.imagelab.issue_scores[self.issue_name].values())
        )
        raw_scores[raw_scores > 1] = 1
        scores = raw_scores
        # scores: np.ndarray = 1 - np.exp(-1 * raw_scores * self.t)color='r'
        self.imagelab.results[f"{self.issue_name} raw_scores"] = raw_scores
        self.imagelab.results[f"{self.issue_name} score"] = scores
        self.imagelab.results[f"{self.issue_name} bool"] = self.mark_bool_issues(
            scores, threshold
        )
        self.num_issues = np.sum(
            self.imagelab.results[f"{self.issue_name} bool"].tolist()
        )

    def visualize(self, num_preview=10):
        results_col = self.imagelab.results[f"{self.issue_name} bool"]
        scores_col = self.imagelab.results[f"{self.issue_name} score"]
        issue_scores = scores_col.to_numpy()
        issue_indices = np.argsort(issue_scores)
        true_issue_indices = set(self.imagelab.results.index[results_col].tolist())
        issue_indices = [idx for idx in issue_indices if idx in true_issue_indices]

        if (
            num_preview is None
        ):  # TODO: kind of strange I think display_images should be rewritten
            num_preview = len(issue_indices)
        for ind in display_images(
            issue_indices, num_preview
        ):  # show the top 10 issue images (if exists)
            img_name = self.imagelab.image_files[ind]
            issue_score = self.imagelab.results[
                self.imagelab.results["image_name"] == img_name
            ][f"{self.issue_name} score"].tolist()[0]
            title = f"{self.issue_name} [{issue_score}] {img_name}"
            try:
                img = Image.open(
                    os.path.join(self.imagelab.path, self.imagelab.image_files[ind])
                )
                print(title)
                img.show(title=title)
            except:
                break


class LightImagesIssueManager(IssueManager):

    # TODO: Add `results_key = "label"` to this class
    # TODO: Add `info_keys = ["label"]` to this class
    def __init__(self, imagelab: Imagelab):
        super().__init__(imagelab)
        self.issue_name = "LightImages"
        self.threshold = 1
        self.t = 1

    def find_issues(self, img, image_name, **kwargs) -> pd.DataFrame:
        score = get_brightness_score(img)
        self.update_info(image_name, score)
        return score

    def update_info(self, image_name, score, **kwargs) -> None:
        self.imagelab.issue_scores[self.issue_name][image_name] = score

    # def mark_bool_issues(self, raw_scores):
    #     return get_is_issue(raw_scores, self.imagelab.thresholds)

    def aggregate(self, threshold):
        raw_scores = np.array(
            list(self.imagelab.issue_scores[self.issue_name].values())
        )
        raw_scores[raw_scores > 1] = 1
        scores = 1 - raw_scores
        # scores: np.ndarray = np.exp(-1 * raw_scores * self.t)
        self.imagelab.results[f"{self.issue_name} score"] = scores
        self.imagelab.results[f"{self.issue_name} bool"] = self.mark_bool_issues(
            scores, threshold
        )
        self.num_issues = np.sum(
            self.imagelab.results[f"{self.issue_name} bool"].tolist()
        )

    def visualize(self, num_preview=10):
        results_col = self.imagelab.results[f"{self.issue_name} bool"]
        scores_col = self.imagelab.results[f"{self.issue_name} score"]
        issue_scores = scores_col.to_numpy()
        issue_indices = np.argsort(issue_scores)
        true_issue_indices = set(self.imagelab.results.index[results_col].tolist())
        issue_indices = [idx for idx in issue_indices if idx in true_issue_indices]

        if (
            num_preview is None
        ):  # TODO: kind of strange I think display_images should be rewritten
            num_preview = len(issue_indices)

        for ind in display_images(
            issue_indices, num_preview
        ):  # show the top 10 issue images (if exists)
            img_name = self.imagelab.image_files[ind]
            issue_score = self.imagelab.results[
                self.imagelab.results["image_name"] == img_name
            ][f"{self.issue_name} score"].tolist()[0]
            title = f"{self.issue_name} [{issue_score}] {img_name}"
            try:
                img = Image.open(
                    os.path.join(self.imagelab.path, self.imagelab.image_files[ind])
                )
                print(title)
                img.show(title=title)
            except:
                break


class BlurredIssueManager(IssueManager):

    # TODO: Add `results_key = "label"` to this class
    # TODO: Add `info_keys = ["label"]` to this class

    def __init__(self, imagelab: Imagelab):
        super().__init__(imagelab)
        self.issue_name = "Blurred"
        self.t = 0.001

    def find_issues(self, img, image_name, **kwargs) -> pd.DataFrame:
        score = check_blurriness(img)
        self.update_info(image_name, score)
        return score

    def update_info(self, image_name, score, **kwargs) -> None:
        self.imagelab.issue_scores[self.issue_name][image_name] = score

    # def mark_bool_issues(self, scores, threshold):
    #     return scores < threshold

    def aggregate(self, threshold):
        raw_scores = np.array(
            list(self.imagelab.issue_scores[self.issue_name].values())
        )
        # self.imagelab.results[f'{self.issue_name} raw_score'] = raw_scores
        scores: np.ndarray = 1 - np.exp(-1 * raw_scores * self.t)
        self.imagelab.results[f"{self.issue_name} score"] = scores
        self.imagelab.results[f"{self.issue_name} bool"] = self.mark_bool_issues(
            scores, threshold
        )
        self.num_issues = np.sum(
            self.imagelab.results[f"{self.issue_name} bool"].tolist()
        )

    def visualize(self, num_preview=10):
        results_col = self.imagelab.results[f"{self.issue_name} bool"]
        scores_col = self.imagelab.results[f"{self.issue_name} score"]
        issue_scores = scores_col.to_numpy()
        issue_indices = np.argsort(issue_scores)
        true_issue_indices = set(self.imagelab.results.index[results_col].tolist())
        issue_indices = [idx for idx in issue_indices if idx in true_issue_indices]

        if (
            num_preview is None
        ):  # TODO: kind of strange I think display_images should be rewritten
            num_preview = len(issue_indices)

        for ind in display_images(
            issue_indices, num_preview
        ):  # show the top 10 issue images (if exists)
            img_name = self.imagelab.image_files[ind]
            issue_score = self.imagelab.results[
                self.imagelab.results["image_name"] == img_name
            ][f"{self.issue_name} score"].tolist()[0]
            title = f"{self.issue_name} [{issue_score}] {img_name}"
            try:
                img = Image.open(os.path.join(self.imagelab.image_files[ind]))
                print(title)
                img.show(title=title)
            except:
                break


class AspectRatioIssueManager(IssueManager):

    # TODO: Add `results_key = "label"` to this class
    # TODO: Add `info_keys = ["label"]` to this class

    def __init__(self, imagelab: Imagelab):
        super().__init__(imagelab)
        self.issue_name = "AspectRatio"
        self.t = 1

    def find_issues(self, img, image_name, **kwargs) -> pd.DataFrame:
        score = check_odd_size(img)
        self.update_info(image_name, score)
        return score

    def update_info(self, image_name, score, **kwargs) -> None:
        self.imagelab.issue_scores[self.issue_name][image_name] = score

    # def mark_bool_issues(self, raw_scores):
    #     return get_is_issue(raw_scores, self.imagelab.thresholds)

    def aggregate(self, threshold):
        raw_scores = np.array(
            list(self.imagelab.issue_scores[self.issue_name].values())
        )
        scores = raw_scores
        # scores: np.ndarray = 1 - np.exp(-1 * raw_scores * self.t)
        # self.imagelab.results[f'{self.issue_name} raw_score'] = raw_scores
        self.imagelab.results[f"{self.issue_name} score"] = scores
        self.imagelab.results[f"{self.issue_name} bool"] = self.mark_bool_issues(
            scores, threshold
        )
        self.num_issues = np.sum(
            self.imagelab.results[f"{self.issue_name} bool"].tolist()
        )

    def visualize(self, num_preview=10):
        results_col = self.imagelab.results[f"{self.issue_name} bool"]
        scores_col = self.imagelab.results[f"{self.issue_name} score"]
        issue_scores = scores_col.to_numpy()
        issue_indices = np.argsort(issue_scores)
        true_issue_indices = set(self.imagelab.results.index[results_col].tolist())
        issue_indices = [idx for idx in issue_indices if idx in true_issue_indices]

        if (
            num_preview is None
        ):  # TODO: kind of strange I think display_images should be rewritten
            num_preview = len(issue_indices)

        for ind in display_images(
            issue_indices, num_preview
        ):  # show the top 10 issue images (if exists)
            img_name = self.imagelab.image_files[ind]
            issue_score = self.imagelab.results[
                self.imagelab.results["image_name"] == img_name
            ][f"{self.issue_name} score"].tolist()[0]
            title = f"{self.issue_name} [{issue_score}] {img_name}"
            try:
                img = Image.open(
                    os.path.join(self.imagelab.path, self.imagelab.image_files[ind])
                )
                print(title)
                img.show(title=title)
            except:
                break


class HotPixelsIssueManager(IssueManager):

    # TODO: Add `results_key = "label"` to this class
    # TODO: Add `info_keys = ["label"]` to this class

    def __init__(self, imagelab: Imagelab):
        super().__init__(imagelab)
        self.issue_name = "HotPixels"
        self.threshold = 1
        self.t = 1

    def find_issues(self, img, image_name, **kwargs) -> pd.DataFrame:
        score = find_hot_pixels(img)
        self.update_info(image_name, score)
        return score

    def update_info(self, image_name, score, **kwargs) -> None:
        self.imagelab.issue_scores[self.issue_name][image_name] = score

    def mark_bool_issues(self, raw_scores, threshold):
        return get_is_issue(raw_scores, self.imagelab.thresholds)

    def aggregate(self, threshold):
        raw_scores = np.array(
            list(self.imagelab.issue_scores[self.issue_name].values())
        )
        scores: np.ndarray = np.exp(-1 * raw_scores * self.t)
        self.imagelab.results[f"{self.issue_name} score"] = scores
        self.imagelab.results[f"{self.issue_name} bool"] = self.mark_bool_issues(
            scores, threshold
        )
        self.num_issues = np.sum(
            self.imagelab.results[f"{self.issue_name} bool"].tolist()
        )

    def visualize(self, num_preview=10):
        results_col = self.imagelab.results[f"{self.issue_name} bool"]
        scores_col = self.imagelab.results[f"{self.issue_name} score"]
        issue_scores = scores_col.to_numpy()
        issue_indices = np.argsort(issue_scores)
        true_issue_indices = set(self.imagelab.results.index[results_col].tolist())
        issue_indices = [idx for idx in issue_indices if idx in true_issue_indices]

        if (
            num_preview is None
        ):  # TODO: kind of strange I think display_images should be rewritten
            num_preview = len(issue_indices)

        for ind in display_images(
            issue_indices, num_preview
        ):  # show the top 10 issue images (if exists)
            img_name = self.imagelab.image_files[ind]
            issue_score = self.imagelab.results[
                self.imagelab.results["image_name"] == img_name
            ][f"{self.issue_name} score"].tolist()[0]
            title = f"{self.issue_name} [{issue_score}] {img_name}"
            try:
                img = Image.open(
                    os.path.join(self.imagelab.path, self.imagelab.image_files[ind])
                )
                print(title)
                img.show(title=title)
            except:
                break


class GrayscaleIssueManager(IssueManager):

    # TODO: Add `results_key = "label"` to this class
    # TODO: Add `info_keys = ["label"]` to this class

    def __init__(self, imagelab: Imagelab):
        super().__init__(imagelab)
        self.issue_name = "Grayscale"
        self.threshold = 1

    def find_issues(self, img, image_name, **kwargs) -> pd.DataFrame:
        score = check_grayscale(img)
        self.update_info(image_name, score)
        return score

    def update_info(self, image_name, score, **kwargs) -> None:
        self.imagelab.issue_scores[self.issue_name][image_name] = score

    # def mark_bool_issues(self, raw_scores):
    #     return raw_scores.astype('bool')

    def aggregate(self, threshold):
        raw_scores = np.array(
            list(self.imagelab.issue_scores[self.issue_name].values())
        )
        # todo this zscore does not make sense, this value can just be used as a score since this is a binary variable
        self.imagelab.results[f"{self.issue_name} score"] = 1 - raw_scores
        self.imagelab.results[f"{self.issue_name} bool"] = self.mark_bool_issues(
            1 - raw_scores, threshold
        )
        self.num_issues = np.sum(
            self.imagelab.results[f"{self.issue_name} bool"].tolist()
        )

    def visualize(self, num_preview=10):
        results_col = self.imagelab.results[f"{self.issue_name} bool"]
        scores_col = self.imagelab.results[f"{self.issue_name} score"]
        issue_scores = scores_col.to_numpy()
        issue_indices = np.argsort(issue_scores)
        true_issue_indices = set(self.imagelab.results.index[results_col].tolist())
        issue_indices = [idx for idx in issue_indices if idx in true_issue_indices]

        if (
            num_preview is None
        ):  # TODO: kind of strange I think display_images should be rewritten
            num_preview = len(issue_indices)

        for ind in display_images(
            issue_indices, num_preview
        ):  # show the top 10 issue images (if exists)
            img_name = self.imagelab.image_files[ind]
            issue_score = self.imagelab.results[
                self.imagelab.results["image_name"] == img_name
            ][f"{self.issue_name} score"].tolist()[0]
            title = f"{self.issue_name} [{issue_score}] {img_name}"
            try:
                img = Image.open(
                    os.path.join(self.imagelab.path, self.imagelab.image_files[ind])
                )
                print(title)
                img.show(title=title)
            except:
                break


# Construct concrete issue manager with a from_str method
class _IssueManagerFactory:
    """Factory class for constructing concrete issue managers."""

    # todo: convert these strings to constants
    types = {
        "Duplicated": DuplicatedIssueManager,
        "AspectRatio": AspectRatioIssueManager,
        "DarkImages": DarkImagesIssueManager,
        "LightImages": LightImagesIssueManager,
        "Blurred": BlurredIssueManager,
        "Entropy": EntropyIssueManager,
        "NearDuplicates": CheckNearDuplicatesIssueManager,
        "Grayscale": GrayscaleIssueManager,
        "HotPixels": HotPixelsIssueManager,
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
