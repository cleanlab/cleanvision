from __future__ import annotations

from typing import List, Optional, Dict

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
        self.metadata["index_to_path"] = pd.DataFrame(
            {"image_path": self._filepaths}, index=self.index
        )

    def __len__(self) -> int:
        return len(self._filepaths)

    def __getitem__(self, item: int) -> Image.Image:
        path = self._filepaths[item]
        return Image.open(path)

    def _set_index(self) -> None:
        self.index = [i for i in range(len(self._filepaths))]

    def get_name(self, item: int) -> str:
        path = self._filepaths[item]
        return path.split("/")[-1]

    def get_index_to_path_mapping(self) -> Dict[int, str]:
        return {i: path for i, path in enumerate(self._filepaths)}
