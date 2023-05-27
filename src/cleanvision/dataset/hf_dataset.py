from __future__ import annotations

from typing import TYPE_CHECKING, Union, Optional

from PIL import Image

from cleanvision.dataset.base_dataset import Dataset

if TYPE_CHECKING:  # pragma: no cover
    import datasets


class HFDataset(Dataset):
    """Wrapper class to handle datasets loaded from Huggingface."""

    def __init__(self, hf_dataset: "datasets.Dataset", image_key: str):
        super().__init__()
        self._data = hf_dataset
        if image_key in hf_dataset.features:
            self._image_key = image_key
        else:
            raise ValueError(
                "Please specify the right image_key for this dataset. It can be found in dataset.features dict."
            )
        self._set_index()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: Union[int, str]) -> Optional[Image.Image]:
        try:
            image = self._data[item][self._image_key]
            return image
        except Exception as e:
            print(f"Could not load image at index: {item}\n", e)
            return None

    def _set_index(self) -> None:
        self.index = [i for i in range(len(self._data))]

    def get_name(self, item: Union[int, str]) -> str:
        return f"idx: {item}"
