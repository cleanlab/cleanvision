from __future__ import annotations

from typing import List, Optional

import pandas as pd
from PIL import Image

from cleanvision.dataset import Dataset
from cleanvision.utils.utils import get_filepaths


class FolderDataset(Dataset):
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
        self.metadata["index_to_path"] = pd.DataFrame(self._filepaths, index=self.index)

    def __len__(self) -> int:
        return len(self._filepaths)

    def __getitem__(self, item: int) -> Image.Image:
        path = self._filepaths[item]
        return Image.open(path)

    def __next__(self) -> Image.Image:
        if self._idx >= len(self._filepaths):
            raise StopIteration
        image = Image.open(self._filepaths[self._idx])
        self._idx += 1
        return image

    def _set_index(self) -> None:
        self.index = [i for i in range(len(self._filepaths))]

    def get_name(self, item: int) -> str:
        path = self._filepaths[item]
        return path.split("/")[-1]
