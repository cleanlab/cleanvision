"""Videolab is an extension of Imagelab for finding issues in a video dataset."""
from __future__ import annotations

import pickle
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Type, TypeVar, Union

import pandas as pd
from PIL.Image import Image
from tqdm.auto import tqdm

from cleanvision.dataset.base_dataset import Dataset
from cleanvision.imagelab import Imagelab
from cleanvision.utils.utils import get_is_issue_colname
from cleanvision.utils.constants import DEFAULT_ISSUE_TYPES_VIDEOLAB
import os

OBJECT_FILENAME = "videolab.pkl"
ISSUES_FILENAME = "frame_issues.csv"
ISSUE_SUMMARY_FILENAME = "frame_issue_summary.csv"
VIDEO_FILE_EXTENSIONS = ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm"]

__all__ = ["Videolab"]
TVideolab = TypeVar("TVideolab", bound="Videolab")


class VideoDataset(Dataset):
    """Wrapper class to handle video datasets."""

    def __init__(
        self,
        data_folder: Optional[str] = None,
        filepaths: Optional[List[str]] = None,
    ) -> None:
        """Determine video dataset source and populate index."""
        # check if data folder is given
        if data_folder:
            # get filepaths from video dataset directory
            self._filepaths = [
                str(path) for path in self.__get_filepaths(Path(data_folder))
            ]

        else:
            # store user supplied video file paths
            assert filepaths is not None
            self._filepaths = filepaths

        # create index
        self._set_index()

    def __len__(self) -> int:
        """Get video dataset file count."""
        return len(self.index)

    def __iter__(self) -> Iterator[Union[int, str]]:
        """Defining the iteration behavior."""
        return iter(self.index)

    def _set_index(self) -> None:
        """Create internal storage for filepaths."""
        self.index = [path for path in self._filepaths]

    def __get_filepaths(self, dataset_path: Path) -> Generator[Path, None, None]:
        """Scan file system for video files and grab their file paths."""
        # notify user
        print(f"Reading videos from {dataset_path}")

        # iterate over video file extensions
        for ext in VIDEO_FILE_EXTENSIONS:
            # loop through video paths matching ext
            yield from dataset_path.glob(f"**/{ext}")


class FrameSampler:
    """Simplest frame sampling strategy."""

    def __init__(self, k: int) -> None:
        """Store frame sample interval k and import PyAV."""
        # storing frame sampling interval
        self.k = k

        # attempting to import PyAV
        try:
            self.av = import_module("av")
        except ImportError as error:
            raise ImportError(
                "Cannot import package `av`. "
                "Please install it via `pip install av` and then try again."
            ) from error

    def _create_frame_sample_sub_dir(self, output_dir: Path, idx: int) -> Path:
        """Create a unique sub direcotry for storing frame samples from a video file."""
        # create new sub directory from video_dataset index
        sub_dir = output_dir / str(idx)
        sub_dir.mkdir(parents=True)

        # return path to new sub dir
        return sub_dir

    def sample(self, video_dataset: VideoDataset, output_dir: Path) -> None:
        """Loop through frames and store every k-th frame."""
        # notify of sampling
        print(f"Sampling frames at every {self.k} frames ...")

        # iterate over video files in video data directory
        for idx, video_file in enumerate(tqdm(video_dataset)):
            # create frame samples sub directory
            sample_sub_dir = self._create_frame_sample_sub_dir(output_dir, idx)

            # open video file for streaming
            with self.av.open(str(video_file)) as container:
                # get video stream
                stream = container.streams.video[0]

                # iterate frames
                for frame_indx, frame in enumerate(container.decode(stream)):
                    # check for k-th frame
                    if not frame_indx % self.k:
                        # get PIL image
                        frame_pil: Image = frame.to_image()

                        # use frame timestamp as image file name
                        image_file_name = str(frame.time) + ".jpg"

                        # save to output dir
                        frame_pil.save(sample_sub_dir / image_file_name)


class Videolab:
    """A single class to find all types of issues in video datasets."""

    def __init__(
        self,
        video_dir: Optional[str] = None,
        video_filepaths: Optional[List[str]] = None,
    ) -> None:
        """Create Path object from video directory string."""
        # store video dataset
        self.video_dataset: VideoDataset = VideoDataset(video_dir, video_filepaths)

    def _sample_frames(self, samples_dir: Path, sample_interval: int) -> None:
        """Get sample frames."""
        # setup frame sampler
        frame_sampler = FrameSampler(sample_interval)

        # sample frames from target video data directory
        frame_sampler.sample(self.video_dataset, samples_dir)

    @staticmethod
    def _parent_dir_frame_samples_dict(
        frame_issues: pd.DataFrame,
    ) -> Dict[str, List[str]]:
        """Creates dictionary of parent directory and frame samples."""
        # set dict
        cluster_frame_samples: Dict[str, List[str]] = {}

        # looper over index
        for img_path in frame_issues.index:
            # get frame sample parent
            sample_dir = Path(img_path).parents[0]

            # get key
            key = str(sample_dir)

            # check if key exists
            if key in cluster_frame_samples:
                # update
                cluster_frame_samples[key].append(img_path)

            else:
                # create new entry
                cluster_frame_samples[key] = [img_path]

        # get cluster dict
        return cluster_frame_samples

    def _aggregate_issues(self, frame_issues: pd.DataFrame) -> pd.DataFrame:
        """Aggregate Imagelab issues into a single frame for each video."""
        # convert booleans to floats
        pure_float_issues = frame_issues * 1

        # store new aggregate_issues
        aggregate_issues = []

        # loop over clusters
        for _, indexes in self._parent_dir_frame_samples_dict(frame_issues).items():
            # get all frame issues for sample_dir subset
            frame_issues = pure_float_issues.loc[indexes]

            # calculate new index
            new_index = indexes[int(len(indexes) / 2)]

            # create aggregated scores df
            aggregate_issues.append(
                pd.DataFrame(frame_issues.mean().to_dict(), index=[new_index])
            )

        # finally create a new DataFrame of all aggregate results
        agg_df = pd.concat(aggregate_issues)

        # create lists of columns
        issue_columns = [
            get_is_issue_colname(issue) for issue in self.imagelab._issue_types
        ]

        # convert float represent average booleans back to booleans
        agg_df[issue_columns] = agg_df[issue_columns].astype(bool)

        # return the aggregated dataframe
        return agg_df

    def _aggregate_summary(self, aggregate_issues: pd.DataFrame) -> pd.DataFrame:
        """Create issues summary for aggregate issues."""
        # setup issue summary storage
        summary_dict = {}

        # loop over issue type
        for issue_type in self.imagelab._issue_types:
            # add individual type summaries
            summary_dict[issue_type] = {
                "num_images": aggregate_issues[get_is_issue_colname(issue_type)].sum()
            }

        # reshape summary dataframe
        agg_summary = pd.DataFrame.from_dict(summary_dict, orient="index")
        agg_summary = agg_summary.reset_index()
        agg_summary = agg_summary.rename(columns={"index": "issue_type"})
        agg_summary = agg_summary.astype({"num_images": int, "issue_type": str})

        # return aggregate summary
        return agg_summary

    @staticmethod
    def list_default_issue_types() -> List[str]:
        """Returns list of the default issue types."""
        return DEFAULT_ISSUE_TYPES_VIDEOLAB

    @staticmethod
    def list_possible_issue_types() -> List[str]:
        """Returns list of all possible issue types including custom issues."""
        return Imagelab.list_possible_issue_types()

    def find_issues(
        self,
        frame_samples_dir: Optional[str] = None,
        frame_sampling_interval: int = 30,
        issue_types: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """Sample frames before calling find_issues and aggregating."""
        if not frame_samples_dir:
            frame_samples_dir = os.path.join(os.getcwd(), "frames")

        # todo: handle case where frame dir already exists

        # create sample frames
        self._sample_frames(Path(frame_samples_dir), frame_sampling_interval)

        # get imagelab instance
        self.imagelab = Imagelab(frame_samples_dir)

        # set default issue types
        setattr(
            self.imagelab, "list_default_issue_types", self.list_default_issue_types
        )

        # use imagelab to find issues in frames
        self.imagelab.find_issues(issue_types, n_jobs, verbose)

        # get frame issues
        self.frame_issues = self.imagelab.issues
        self.frame_issue_summary = self.imagelab.issue_summary

        # update aggregate issues/summary
        self.imagelab.issues = self._aggregate_issues(self.frame_issues)
        self.imagelab.issue_summary = self._aggregate_summary(self.imagelab.issues)

    def report(
        self,
        issue_types: Optional[List[str]] = None,
        max_prevalence: Optional[float] = None,
        num_images: Optional[int] = None,
        verbosity: int = 1,
        print_summary: bool = True,
        show_id: bool = False,
    ) -> None:
        """Prints summary of the aggregate issues found in your dataset."""
        # report on video frame samples
        self.imagelab.report(
            issue_types,
            max_prevalence,
            num_images,
            verbosity,
            print_summary,
            show_id,
        )

    def get_stats(self) -> Any:
        """Returns dict of statistics computed from video frames."""
        return self.imagelab.info["statistics"]

    def save(self, path: str, force: bool = False) -> None:
        """Saves this Videolab instance."""
        # get pathlib Path object
        root_save_path = Path(path)

        # check if videolab root save path exists
        if not root_save_path.exists():
            # create otherwise
            root_save_path.mkdir(parents=True, exist_ok=True)
        else:
            if not force:
                raise FileExistsError("Please specify a new path or set force=True")
            print(
                "WARNING: Existing files will be overwritten "
                f"by newly saved files at: {root_save_path}"
            )

        # create specific imagelab sub directory
        imagelab_sub_dir = str(root_save_path / "imagelab")

        # now save imagelab to subdir
        self.imagelab.save(imagelab_sub_dir, force)

        # save aggregate dataframes
        self.frame_issues.to_csv(root_save_path / ISSUES_FILENAME)
        self.frame_issue_summary.to_csv(root_save_path / ISSUE_SUMMARY_FILENAME)

        # copy videolab object
        videolab_copy = deepcopy(self)

        # clear out dataframes
        videolab_copy.frame_issues = None
        videolab_copy.frame_issue_summary = None

        # Save the imagelab object to disk.
        with open(root_save_path / OBJECT_FILENAME, "wb") as f:
            pickle.dump(videolab_copy, f)

        print(f"Saved Videolab to folder: {root_save_path}")
        print(
            "The data path and dataset must be not be changed to maintain consistent "
            "state when loading this Videolab"
        )

    @classmethod
    def load(cls: Type[TVideolab], path: str) -> Videolab:
        """Loads Videolab from given path."""
        # get pathlib Path object
        root_save_path = Path(path)

        # check if path exists
        if not root_save_path.exists():
            raise ValueError(f"No folder found at specified path: {path}")

        with open(root_save_path / OBJECT_FILENAME, "rb") as f:
            videolab: Videolab = pickle.load(f)

        # Load the issues from disk.
        videolab.frame_issues = pd.read_csv(
            root_save_path / ISSUES_FILENAME, index_col=0
        )
        videolab.frame_issue_summary = pd.read_csv(
            root_save_path / ISSUE_SUMMARY_FILENAME, index_col=0
        )

        # create specific imagelab sub directory
        imagelab_sub_dir = str(root_save_path / "imagelab")

        # store imagelab object
        videolab.imagelab = Imagelab.load(imagelab_sub_dir)

        # notify user
        print("Successfully loaded Videolab")

        return videolab
