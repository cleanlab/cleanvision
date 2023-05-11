from __future__ import annotations

from typing import List, Optional, Union

from PIL import Image

from cleanvision.utils.constants import IMAGE_FILE_EXTENSIONS
from cleanvision.dataset.base_dataset import Dataset
import fsspec
import os


class FSDataset(Dataset):
    """Wrapper class to handle datasets loaded from a cloud-based data folder"""

    def __init__(
        self, data_folder: Optional[str] = None, filepaths: Optional[List[str]] = None
    ) -> None:
        super().__init__()
        if data_folder:
            # See: https://filesystem-spec.readthedocs.io/en/latest/api.html#other-known-implementations
            # contains a list of known implementations that may resolve through that url
            # they require adequate module to be installed
            self.fs, dataset_path = fsspec.core.url_to_fs(data_folder)
            self._filepaths = self.__get_filepaths(dataset_path)
        else:
            assert filepaths is not None
            self._filepaths = filepaths
            # here we assume all of the provided filepaths are from the same filesystem
            self.fs, _ = fsspec.core.url_to_fs(self._filepaths[0])
        self._set_index()

    def __len__(self) -> int:
        return len(self._filepaths)

    def __getitem__(self, item: Union[int, str]) -> Image.Image:
        with self.fs.open(item, "rb") as f:
            # avoid ops on the closed file, make a copy
            data = Image.open(f).copy()
        return data

    def _set_index(self) -> None:
        self.index = [path for path in self._filepaths]

    def get_name(self, item: Union[int, str]) -> str:
        assert isinstance(item, str)
        return item.split("/")[-1]

    def __get_filepaths(self, dataset_path: str) -> List[str]:
        """See an issue here: https://github.com/fsspec/filesystem_spec/issues/1019
        There's a problem with proper patterning on /**/ in fsspec"""
        print(f"Reading images from {dataset_path}")
        filepaths = []
        for ext in IMAGE_FILE_EXTENSIONS:
            # initial *.ext search, top level
            path_top_level = os.path.join(dataset_path, ext)
            # lower depths
            path_lower_level = os.path.join(dataset_path, "**", ext)
            for fs_path in (path_top_level, path_lower_level):
                filetype_images = self.fs.glob(fs_path)
                if len(filetype_images) == 0:
                    continue
                filepaths += filetype_images
        unique_filepaths = list(set(filepaths))
        return sorted(
            unique_filepaths
        )  # sort image names alphabetically and numerically
