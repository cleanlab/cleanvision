from cleanvision.dataset.base_dataset import Dataset
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Union
from cleanvision.utils.constants import VIDEO_FILE_EXTENSIONS


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
            # todo: raise an exception if assert fails
            assert filepaths is not None
            self._filepaths = filepaths

        # create index
        self._set_index()
        self.frames_dir = Path.cwd() / "frames"

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

    def __getitem__(self, item: int) -> str:
        return self.index[item]

    def set_frames_dir(self, frames_dir: Path):
        self.frames_dir = frames_dir
