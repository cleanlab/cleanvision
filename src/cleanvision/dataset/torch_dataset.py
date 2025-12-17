from __future__ import annotations

from typing import TYPE_CHECKING, Union

from PIL import Image, UnidentifiedImageError

from cleanvision.dataset.base_dataset import Dataset

if TYPE_CHECKING:  # pragma: no cover
    from torchvision.datasets.vision import VisionDataset


class TorchDataset(Dataset):
    """Wrapper class to handle datasets loaded from torchvision."""

    def __init__(self, torch_dataset: "VisionDataset") -> None:
        super().__init__()
        self._data = torch_dataset
        # todo: catch errors
        self._image_idx = None
        for i, obj in enumerate(torch_dataset[0]):
            if isinstance(obj, Image.Image):
                self._image_idx = i

        if self._image_idx is None:
            raise ValueError("No PIL image found in torchvision dataset sample")

        self._set_index()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: Union[int, str]) -> Image.Image:
        img: Image.Image = self._data[item][self._image_idx]
        return img

    def get_name(self, index: Union[int, str]) -> str:
        return f"idx: {index}"

    def _set_index(self) -> None:
        valid_indices = []
        for i in range(len(self._data)):
            try:
                img = self._data[i][self._image_idx]
                if not isinstance(img, Image.Image):
                    raise UnidentifiedImageError
                valid_indices.append(i)
            except (UnidentifiedImageError, OSError, ValueError, TypeError):
                print(f"Warning: Skipping corrupted image at index {i}")
                continue
        # self.index = [i for i in range(len(self._data))]
        self.index = valid_indices
