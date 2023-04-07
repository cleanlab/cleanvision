from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

from cleanvision.dataset import Dataset

if TYPE_CHECKING:  # pragma: no cover
    from torchvision.datasets.vision import VisionDataset


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
