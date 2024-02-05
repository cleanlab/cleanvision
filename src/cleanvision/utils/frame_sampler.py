from importlib import import_module
from pathlib import Path


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

    def sample(self, video_path: str, output_dir: Path) -> None:
        """Loop through frames and store every k-th frame."""
        with self.av.open(video_path) as container:
            # get video stream
            stream = container.streams.video[0]

            # iterate frames
            for frame_indx, frame in enumerate(container.decode(stream)):
                # check for k-th frame
                if not frame_indx % self.k:
                    # get PIL image
                    frame_pil = frame.to_image()

                    # use frame timestamp as image file name
                    image_file_name = str(frame.time) + ".jpg"

                    # save to output dir
                    frame_pil.save(output_dir / image_file_name)
