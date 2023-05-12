from __future__ import annotations

from typing import TYPE_CHECKING, Union

from PIL import Image

from cleanvision.dataset.base_dataset import Dataset

if TYPE_CHECKING:  # pragma: no cover
    from torchvision.datasets.vision import VisionDataset


class TorchDataset(Dataset):
    """Wrapper class to handle datasets loaded from torchvision."""

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

    def __getitem__(self, item: Union[int, str]) -> Image.Image:
        return self._data[item][self._image_idx]

    def get_name(self, index: Union[int, str]) -> str:
        return f"idx: {index}"

    def _set_index(self) -> None:
        self.index = [i for i in range(len(self._data))]
