from __future__ import annotations

from collections.abc import Sized
from typing import List, Union, Dict

from PIL import Image


class Dataset(Sized):
    """Wrapper class to handle datasets loaded from various sources like: image files in a local folder, huggingface, or torchvision."""

    def __init__(self) -> None:
        self.index: List[Union[int, str]] = []
        self.metadata: Dict[str, Union[List[str]]] = {}

    def _set_index(self) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, item: Union[int, str]) -> Image.Image:
        raise NotImplementedError

    def get_name(self, index: Union[int, str]) -> str:
        raise NotImplementedError

    def get_index_to_path_mapping(self) -> Dict[int, str]:
        raise NotImplementedError
