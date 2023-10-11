"""Videolab is an extension of Imagelab for finding issues in a video dataset."""
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Generator
from typing import Optional

import av
from cleanvision.imagelab import Imagelab
from PIL.Image import Image


VIDEO_FILE_EXTENSIONS = ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm"]


class FrameSampler:
    """Simplest frame sampling strategy."""

    def __init__(self, k: int) -> None:
        """Store frame sample interval k."""
        self.k = k

    def _create_frame_sample_sub_dir(self, video_file: Path, output_dir: Path) -> Path:
        """Create a unique sub direcotry for storing frame samples from a video file."""
        # create new sub directory from video file name
        sub_dir = output_dir / video_file.name
        sub_dir.mkdir(parents=True)

        # return path to new sub dir
        return sub_dir

    def sample(self, video_file: Path, output_dir: Path) -> None:
        """Loop through frames and store every k-th frame."""
        # create frame samples sub directory
        sample_sub_dir = self._create_frame_sample_sub_dir(video_file, output_dir)

        # open video file for streaming
        with av.open(str(video_file)) as container:
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
        video_dir: str,
    ) -> None:
        """Create Path object from video directory string."""
        self.video_dir: Path = Path(video_dir)
        self.imagelab: Optional[Imagelab] = None

    def _find_videos(self) -> Generator[Path, None, None]:
        """Iterate over video files in video directory."""
        # iterate over video file extensions
        for ext in VIDEO_FILE_EXTENSIONS:
            # loop through video paths matching ext
            yield from self.video_dir.glob(f"**/{ext}")

    def _sample_frames(self, samples_dir: Path, sample_interval: int) -> None:
        """Get sample frames."""
        # setup frame sampler
        frame_sampler = FrameSampler(sample_interval)

        # iterate over video files in video data directory
        for video_file in self._find_videos():
            # sample frames from target video data directory
            frame_sampler.sample(video_file, samples_dir)

    def find_issues(
        self,
        frame_samples_dir: str,
        frame_samples_interval: int,
        issue_types: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """Sampe frames before call Imagelab.find_issues."""
        # create sample frames
        self._sample_frames(Path(frame_samples_dir), frame_samples_interval)

        # create Imagelab instance
        self.imagelab = Imagelab(frame_samples_dir)

        # call Imagelab to find issues in sampled frames
        self.imagelab.find_issues(issue_types, n_jobs, verbose)

    def report(
        self,
        issue_types: Optional[List[str]] = None,
        max_prevalence: Optional[float] = None,
        num_images: Optional[int] = None,
        verbosity: int = 1,
        print_summary: bool = True,
        show_id: bool = False,
    ) -> None:
        """Prints summary of the issues found in your dataset."""
        # check if imagelab instance exists
        if self.imagelab is None:
            print(
                "Please specify some issue_types to"
                "check for in videolab.find_issues()."
            )

        else:
            # report on video frame samples
            self.imagelab.report(
                issue_types,
                max_prevalence,
                num_images,
                verbosity,
                print_summary,
                show_id,
            )
