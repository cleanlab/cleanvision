from __future__ import annotations

from collections.abc import Sized
from typing import List, Union, Dict

from PIL import Image


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
