from typing import TYPE_CHECKING, List, TypeVar, Optional, Self

from PIL import Image

from cleanvision.utils.utils import get_filepaths

if TYPE_CHECKING:
    import datasets
    import torch

TDataset = TypeVar("TDataset", bound="Dataset")


class Dataset:
    """This class is used for managing different kinds of data formats provided by user"""

    def __init__(self) -> None:
        self._set_index()
        self.metadata = {}

    def _set_index(self) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, item: int) -> Image.Image:
        raise NotImplementedError

    def __iter__(self) -> Self:
        self._idx = 0
        return self

    def __next__(self) -> Image.Image:
        raise NotImplementedError

    def get_name(self, index: int) -> str:
        raise NotImplementedError


class HFDataset(Dataset):
    def __init__(self, hf_dataset: datasets.Dataset, image_key: str):
        super().__init__()
        self._data = hf_dataset
        self._image_key = image_key

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
        assert data_folder is not None or filepaths is not None
        self._filepaths = get_filepaths(data_folder) if data_folder else filepaths
        self.metadata["path"] = self._filepaths.copy()

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
    def __init__(self, torch_dataset: torch.utils.data.Dataset) -> None:
        super().__init__()
        self._data = torch_dataset
        # todo: catch errors
        for i, obj in enumerate(torch_dataset[0]):
            if isinstance(obj, Image.Image):
                self._image_idx = i

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
