from typing import TYPE_CHECKING, Union, Tuple, List, TypeVar

from PIL import Image

from cleanvision.utils.utils import get_filepaths

if TYPE_CHECKING:
    import datasets
    import torch

TDataset = TypeVar("TDataset", bound="Dataset")


class Dataset:
    """This class is used for managing different kinds of data formats provided by user"""

    def __init__(self) -> None:
        self.index_list = None

    def _set_index(self) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: Union[str, int]) -> Image.Image:
        raise NotImplementedError

    def __iter__(self) -> TDataset:
        self._idx = 0
        return self

    def __next__(self) -> Tuple[Union[str, int], Image.Image]:
        raise NotImplementedError

    def get_name(self, index: Union[str, int]) -> str:
        raise NotImplementedError


class HFDataset(Dataset):
    def __init__(self, hf_dataset: datasets.Dataset, image_key: str):
        super().__init__()
        self._data = hf_dataset
        self._image_key = image_key
        self._set_index()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: Union[str, int]) -> Image.Image:
        return self._data[index][self._image_key]

    def __next__(self) -> Tuple[int, Image.Image]:
        if self._idx >= len(self._data):
            raise StopIteration
        # todo : catch reading image errors
        index, image = self._idx, self._data[self._idx][self._image_key]
        self._idx += 1
        return index, image

    def _set_index(self) -> None:
        self.index_list: List[Union[str, int]] = [i for i in range(len(self._data))]

    def get_name(self, index: Union[str, int]) -> str:
        return f"idx: {index}"


class FolderDataset(Dataset):
    def __init__(self, data_folder: str) -> None:
        super().__init__()
        self._filepaths = get_filepaths(data_folder)
        self._set_index()

    def __len__(self) -> int:
        return len(self._filepaths)

    def __getitem__(self, index: Union[str, int]) -> Image.Image:
        return Image.open(index)

    def __next__(self) -> Tuple[str, Image.Image]:
        if self._idx >= len(self._filepaths):
            raise StopIteration
        index, image = self._filepaths[self._idx], Image.open(
            self._filepaths[self._idx]
        )
        self._idx += 1
        return index, image

    def _set_index(self) -> None:
        self.index_list = self._filepaths.copy()

    def get_name(self, index: Union[str, int]) -> str:
        return index.split("/")[-1]


class FilePathDataset(Dataset):
    def __init__(self, filepaths: List[str]) -> None:
        super().__init__()
        self._filepaths = filepaths
        self._set_index()

    def __len__(self) -> int:
        return len(self._filepaths)

    def __getitem__(self, index: Union[str, int]) -> Image.Image:
        return Image.open(index)

    def __next__(self) -> Tuple[str, Image.Image]:
        if self._idx >= len(self._filepaths):
            raise StopIteration
        index, image = self._filepaths[self._idx], Image.open(
            self._filepaths[self._idx]
        )
        self._idx += 1
        return index, image

    def _set_index(self) -> None:
        self.index_list = self._filepaths.copy()


# todo
class TorchDataset(Dataset):
    def __init__(self, torch_dataset: torch.utils.data.Dataset) -> None:
        super().__init__()
        self._data = torch_dataset
        # todo: catch errors
        for i, obj in enumerate(torch_dataset[0]):
            if isinstance(obj, Image.Image):
                self._image_idx = i
        self._set_index()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: Union[str, int]) -> Image.Image:
        return self._data[index][self._image_idx]

    def __next__(self) -> Tuple[int, Image.Image]:
        if self._idx >= len(self._data):
            raise StopIteration
        # todo : catch reading image errors
        index, image = self._idx, self._data[self._idx][self._image_idx]
        self._idx += 1
        return index, image

    def get_name(self, index: Union[str, int]) -> str:
        return f"idx: {index}"

    def _set_index(self) -> None:
        self.index_list = [i for i in range(len(self._data))]
