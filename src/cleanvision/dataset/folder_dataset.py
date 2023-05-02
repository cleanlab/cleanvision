from __future__ import annotations

from typing import List, Optional, Union

from PIL import Image

from cleanvision.dataset.base_dataset import Dataset
from cleanvision.utils.utils import get_filepaths


class FolderDataset(Dataset):
    """Wrapper class to handle datasets loaded from a local data folder"""

    def __init__(
        self, data_folder: Optional[str] = None, filepaths: Optional[List[str]] = None
    ) -> None:
        super().__init__()
        if data_folder:
            self._filepaths = get_filepaths(data_folder)
        else:
            assert filepaths is not None
            self._filepaths = filepaths
        self._set_index()

    def __len__(self) -> int:
        return len(self._filepaths)

    def __getitem__(self, item: Union[int, str]) -> Image.Image:
        return Image.open(item)

    def _set_index(self) -> None:
        self.index = [path for path in self._filepaths]

    def get_name(self, item: Union[int, str]) -> str:
        assert isinstance(item, str)
        return item.split("/")[-1]
