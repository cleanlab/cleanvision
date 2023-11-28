"""Videolab is an extension of Imagelab for finding issues in a video dataset."""
from __future__ import annotations


import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar
from cleanvision.utils.frame_sampler import FrameSampler
import pandas as pd
from tqdm.auto import tqdm

from cleanvision.imagelab import Imagelab
from cleanvision.utils.utils import get_is_issue_colname
from cleanvision.utils.constants import DEFAULT_ISSUE_TYPES_VIDEOLAB
from cleanvision.dataset.video_dataset import VideoDataset


OBJECT_FILENAME = "videolab.pkl"
ISSUES_FILENAME = "frame_issues.csv"
ISSUE_SUMMARY_FILENAME = "frame_issue_summary.csv"


__all__ = ["Videolab"]
TVideolab = TypeVar("TVideolab", bound="Videolab")


# todo: should work on a single video file rather than a dataset


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

    def _sample_frames(self, video_dataset, sample_interval: int) -> None:
        """Get sample frames."""
        # setup frame sampler
        frame_sampler = FrameSampler(sample_interval)

        # sample frames from target video data directory
        for idx, path in enumerate(tqdm(video_dataset)):
            output_dir = video_dataset.frames_dir / str(idx)
            output_dir.mkdir(parents=True)
            frame_sampler.sample(path, output_dir)

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
        frame_issues = frame_issues * 1
        frame_issues["video_path"] = [
            self.video_dataset[int(idx.split("/")[-2])]
            for idx in frame_issues.index.tolist()
        ]

        agg_df = frame_issues.groupby("video_path").agg("mean")
        agg_df = agg_df.astype(
            {
                get_is_issue_colname(issue_type): "bool"
                for issue_type in self.issue_types
            }
        )

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
        frames_dir: Optional[str] = None,
        frame_sampling_interval: int = 30,
        issue_types: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """Sample frames before calling find_issues and aggregating."""
        if frames_dir:
            self.video_dataset.set_frames_dir(Path(frames_dir))

        # todo: implement reusing frames across runs
        if self.video_dataset.frames_dir.is_dir():
            raise ValueError(
                "Frames dir already exists. Please provide a new path or delete existing path."
            )

        # create sample frames
        self._sample_frames(self.video_dataset, frame_sampling_interval)

        # get imagelab instance

        self.imagelab = Imagelab(data_path=str(self.video_dataset.frames_dir))

        if not issue_types:
            self.issue_types = {
                issue_type: {} for issue_type in self.list_default_issue_types()
            }
        else:
            self.issue_types = issue_types

        # use imagelab to find issues in frames
        self.imagelab.find_issues(
            issue_types=self.issue_types, n_jobs=n_jobs, verbose=verbose
        )

        # get frame issues
        self._frame_level_issues = self.imagelab.issues

        # update aggregate issues/summary

        # todo: change aggregate method to use video path as index

        self.issues = self._aggregate_issues(self._frame_level_issues.copy())
        self.issue_summary = pd.DataFrame(
            [
                {
                    "issue_type": issue_type,
                    "num_videos": self.issues[get_is_issue_colname(issue_type)].sum(),
                }
                for issue_type in self.issue_types
            ]
        )

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
        self._frame_level_issues.to_csv(root_save_path / ISSUES_FILENAME)
        self.frame_issue_summary.to_csv(root_save_path / ISSUE_SUMMARY_FILENAME)

        # copy videolab object
        videolab_copy = deepcopy(self)

        # clear out dataframes
        videolab_copy._frame_level_issues = None
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
        videolab._frame_level_issues = pd.read_csv(
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
