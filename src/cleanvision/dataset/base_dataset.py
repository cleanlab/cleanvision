from __future__ import annotations

from collections.abc import Sized
from typing import List, Union

from PIL import Image


class Dataset(Sized):
    """Wrapper class to handle datasets loaded from various sources like: image files in a local folder, huggingface, or torchvision."""

    def __init__(self) -> None:
        self.index: List[Union[int, str]] = []

    def _set_index(self) -> None:
        """Sets the `index` attribute of the `Dataset` object."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Returns the number of examples in the dataset"""
        raise NotImplementedError

    def __getitem__(self, item: Union[int, str]) -> Image.Image:
        """Returns the image at a given index"""
        raise NotImplementedError

    def get_name(self, index: Union[int, str]) -> str:
        """Returns the name of the image in the dataset. It can be a filename or a str with index information."""
        raise NotImplementedError
