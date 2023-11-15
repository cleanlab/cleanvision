"""Videolab is an extension of Imagelab for finding issues in a video dataset."""
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Union

import pandas as pd
from PIL.Image import Image
from tqdm.auto import tqdm

from cleanvision.dataset.base_dataset import Dataset
from cleanvision.imagelab import Imagelab
from cleanvision.utils.utils import get_is_issue_colname

VIDEO_FILE_EXTENSIONS = ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm"]


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

    def _parent_dir_frame_samples_dict(self) -> Dict[str, List[str]]:
        """Creates dictionary of parent directory and frame samples."""
        # set dict
        cluster_frame_samples: Dict[str, List[str]] = {}

        # looper over index
        for img_path in self.imagelab.issues.index:
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

    def _aggregate_issues(self) -> pd.DataFrame:
        """Aggregate Imagelab issues into a single frame for each video."""
        # convert booleans to floats
        pure_float_issues = self.imagelab.issues * 1

        # store new aggregate_issues
        aggregate_issues = []

        # loop over clusters
        for _, indexes in self._parent_dir_frame_samples_dict().items():
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

    def _aggregate_summary(self) -> pd.DataFrame:
        """Create issues summary for aggregate issues."""
        # setup issue summary storage
        summary_dict = {}

        # loop over issue type
        for issue_type in self.imagelab._issue_types:
            # add individual type summaries
            summary_dict[issue_type] = {
                "num_images": self.agg_issues[get_is_issue_colname(issue_type)].sum()
            }

        # reshape summary dataframe
        agg_summary = pd.DataFrame.from_dict(summary_dict, orient="index")
        agg_summary = agg_summary.reset_index()
        agg_summary = agg_summary.rename(columns={"index": "issue_type"})
        agg_summary = agg_summary.astype({"num_images": int, "issue_type": str})

        # return aggregate summary
        return agg_summary

    def find_issues(
        self,
        frame_samples_dir: str,
        frame_samples_interval: int,
        issue_types: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """Sample frames before calling find_issues and aggregating."""
        # create sample frames
        self._sample_frames(Path(frame_samples_dir), frame_samples_interval)

        # get imagelab instance
        self.imagelab = Imagelab(frame_samples_dir)

        # use imagelab to find issues in frames
        self.imagelab.find_issues(issue_types, n_jobs, verbose)

        # update aggregate issues/summary
        self.agg_issues = self._aggregate_issues()
        self.agg_summary = self._aggregate_summary()

    def _aggregate_report(
        self,
        issue_types: Optional[List[str]] = None,
        max_prevalence: Optional[float] = None,
        num_images: Optional[int] = None,
        verbosity: int = 1,
        print_summary: bool = True,
        show_id: bool = False,
    ) -> None:
        """Create report visualization for aggregate issues."""
        assert isinstance(verbosity, int) and 0 <= verbosity < 5

        user_supplied_args = locals()
        report_args = self.imagelab._get_report_args(user_supplied_args)

        issue_types_to_report = (
            issue_types if issue_types else self.agg_summary["issue_type"].tolist()
        )

        # filter issues based on max_prevalence in the dataset
        filtered_issue_types = self.imagelab._filter_report(
            issue_types_to_report, report_args["max_prevalence"]
        )

        issue_summary = self.agg_summary[
            self.agg_summary["issue_type"].isin(filtered_issue_types)
        ]
        if len(issue_summary) > 0:
            if verbosity:
                print("Issues found in videos in order of severity in the dataset\n")
            if print_summary:
                self.imagelab._pprint_issue_summary(issue_summary)
            for issue_type in filtered_issue_types:
                if (
                    self.agg_summary.query(f"issue_type == {issue_type!r}")[
                        "num_images"
                    ].values[0]
                    == 0
                ):
                    continue
                print(f"{' ' + issue_type + ' frames ':-^60}\n")
                print(
                    f"Number of examples with this issue: "
                    f"{self.agg_issues[get_is_issue_colname(issue_type)].sum()}\n"
                    f"Examples representing most severe instances of this issue:\n"
                )
                self.imagelab._visualize(
                    issue_type,
                    report_args["num_images"],
                    report_args["cell_size"],
                    show_id,
                )
        else:
            print(
                "Please specify some issue_types to "
                "check for in videolab.find_issues()."
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
        self._aggregate_report(
            issue_types,
            max_prevalence,
            num_images,
            verbosity,
            print_summary,
            show_id,
        )
