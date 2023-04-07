from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from PIL import Image

from cleanvision.dataset import Dataset

if TYPE_CHECKING:  # pragma: no cover
    import datasets


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

    def _set_index(self) -> None:
        self.index = [i for i in range(len(self._data))]

    def get_name(self, item: int) -> str:
        return f"idx: {item}"

    def get_index_to_path_mapping(self) -> Dict[int, str]:
        raise ValueError("Index to path mapping does not exist for this dataset.")
