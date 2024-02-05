"""Videolab is an extension of Imagelab for finding issues in a video dataset."""
from __future__ import annotations


from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar
from cleanvision.utils.frame_sampler import FrameSampler
import pandas as pd
from tqdm.auto import tqdm

from cleanvision.imagelab import Imagelab
from cleanvision.utils.utils import get_is_issue_colname
from cleanvision.utils.constants import DEFAULT_ISSUE_TYPES_VIDEOLAB
from cleanvision.dataset.video_dataset import VideoDataset
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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
            issue_types=self.issue_types, n_jobs=n_jobs, verbose=False
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

    def _pprint_issue_summary(self, issue_summary: pd.DataFrame) -> None:
        issue_summary_copy = issue_summary.copy()
        issue_summary_copy.dropna(axis=1, how="all", inplace=True)
        issue_summary_copy.fillna("N/A", inplace=True)
        print(issue_summary_copy.to_markdown(), "\n")

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

        issue_types_to_report = (
            issue_types if issue_types else self.issue_summary["issue_type"].tolist()
        )

        issue_summary = self.issue_summary[
            self.issue_summary["issue_type"].isin(issue_types_to_report)
        ]
        if len(issue_summary) > 0:
            if verbosity:
                print("Issues found in videos in order of severity in the dataset\n")
            if print_summary:
                self._pprint_issue_summary(issue_summary)
            for issue_type in issue_types_to_report:
                if (
                    self.issue_summary.query(f"issue_type == '{issue_type}'")[
                        "num_videos"
                    ].values[0]
                    == 0
                ):
                    continue
                print(f"{' ' + issue_type + ' videos ':-^60}\n")
                print(
                    f"Number of examples with this issue: {self.issues[get_is_issue_colname(issue_type)].sum()}\n"
                    f"Examples representing most severe instances of this issue:\n"
                )
                # self._visualize(
                #     issue_type,
                #     report_args["num_images"],
                #     report_args["cell_size"],
                #     show_id,
                # )
        else:
            print(
                "Please specify some issue_types to check for in videolab.find_issues()."
            )

    def visualize(self, issue_types):
        for issue_type in issue_types:
            colname = get_is_issue_colname(issue_type)
            video_paths = self.issues[self.issues[colname] is True].index.tolist()
            self.rev_idx = {path: i for i, path in enumerate(self.video_dataset.index)}

            frames_list = []
            for path in video_paths:
                frame_dir = self.video_dataset.frames_dir / str(self.rev_idx[path])
                frames_list.append(
                    [Image.open(frame_path) for frame_path in frame_dir.iterdir()]
                )

            # Define the number of animations and frames
            num_animations = min(4, len(video_paths))
            num_frames = 100

            # Create a figure with subplots
            fig, axes = plt.subplots(1, num_animations, figsize=(8, 8))

            # Initialize data for each animation
            initial_data = [frames[0] for frames in frames_list]
            images = frames_list

            # Function to initialize the subplots
            def init():
                for ax, data in zip(axes, initial_data):
                    ax.imshow(data)
                    ax.axis("off")
                return axes

            # Function to update the subplots for each frame
            def update(frame):
                for ax, image in zip(axes, images):
                    ax.imshow(image[frame])
                return axes

            FuncAnimation(
                fig, update, frames=num_frames, init_func=init, blit=True, interval=100
            )

            plt.tight_layout()
        plt.show()
