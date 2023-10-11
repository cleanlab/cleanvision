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


class Videolab(Imagelab):
    pass
