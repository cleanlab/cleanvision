from __future__ import annotations

from collections.abc import Sized
from typing import TYPE_CHECKING, List, Optional, Union, Dict

import pandas as pd
from PIL import Image

from cleanvision.utils.utils import get_filepaths

if TYPE_CHECKING:  # pragma: no cover
    import datasets
    from torchvision.datasets.vision import VisionDataset


class Dataset(Sized):
    """Wrapper class to handle datasets loaded from various sources like: image files in a local folder, huggingface, or torchvision."""

    def __init__(self) -> None:
        self.index: List[int] = []
        self.metadata: Dict[str, Union[List[str]]] = {}

    def _set_index(self) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, item: int) -> Image.Image:
        raise NotImplementedError

    def __iter__(self) -> Dataset:
        self._idx = 0
        return self

    def __next__(self) -> Image.Image:
        raise NotImplementedError

    def get_name(self, index: int) -> str:
        raise NotImplementedError


class HFDataset(Dataset):
    def __init__(self, hf_dataset: "datasets.Dataset", image_key: str):
        super().__init__()
        self._data = hf_dataset
        self._image_key = image_key
        self._set_index()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: int) -> Image.Image:
        return self._data[item][self._image_key]

    def __next__(self) -> Image.Image:
        if self._idx >= len(self._data):
            raise StopIteration
        # todo : catch reading image errors
        image = self._data[self._idx][self._image_key]
        self._idx += 1
        return image

    def _set_index(self) -> None:
        self.index = [i for i in range(len(self._data))]

    def get_name(self, item: int) -> str:
        return f"idx: {item}"


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


class TorchDataset(Dataset):
    def __init__(self, torch_dataset: "VisionDataset") -> None:
        super().__init__()
        self._data = torch_dataset
        # todo: catch errors
        for i, obj in enumerate(torch_dataset[0]):
            if isinstance(obj, Image.Image):
                self._image_idx = i
        self._set_index()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: int) -> Image.Image:
        return self._data[item][self._image_idx]

    def __next__(self) -> Image.Image:
        if self._idx >= len(self._data):
            raise StopIteration
        # todo : catch reading image errors
        image = self._data[self._idx][self._image_idx]
        self._idx += 1
        return image

    def get_name(self, index: int) -> str:
        return f"idx: {index}"

    def _set_index(self) -> None:
        self.index = [i for i in range(len(self._data))]


def build_dataset(
    data_path: Optional[str] = None,
    filepaths: Optional[List[str]] = None,
    hf_dataset: Optional["datasets.Dataset"] = None,
    image_key: Optional[str] = None,
    torchvision_dataset: Optional["VisionDataset"] = None,
) -> Dataset:
    if data_path:
        return FolderDataset(data_folder=data_path)
    elif filepaths:
        return FolderDataset(filepaths=filepaths)
    elif hf_dataset and image_key:
        return HFDataset(hf_dataset, image_key)
    elif torchvision_dataset:
        return TorchDataset(torchvision_dataset)
    else:
        raise ValueError(
            "Please specify one of data_path, filepaths, (hf_dataset, image_key) or torchvision_dataset to check for issues."
        )
