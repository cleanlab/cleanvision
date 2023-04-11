from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from cleanvision.dataset.base_dataset import Dataset
from cleanvision.dataset.folder_dataset import FolderDataset
from cleanvision.dataset.hf_dataset import HFDataset
from cleanvision.dataset.torch_dataset import TorchDataset

if TYPE_CHECKING:  # pragma: no cover
    import datasets
    from torchvision.datasets.vision import VisionDataset


def build_dataset(
    data_path: Optional[str] = None,
    filepaths: Optional[List[str]] = None,
    hf_dataset: Optional["datasets.Dataset"] = None,
    image_key: Optional[str] = None,
    torchvision_dataset: Optional["VisionDataset"] = None,
) -> Dataset:
    if data_path:
        return FolderDataset(data_folder=data_path)
    elif filepaths:
        return FolderDataset(filepaths=filepaths)
    elif hf_dataset and image_key:
        return HFDataset(hf_dataset, image_key)
    elif torchvision_dataset:
        return TorchDataset(torchvision_dataset)
    else:
        raise ValueError(
            "You must specify some argument among the following: `data_path`, `filepaths`, (`hf_dataset`, `image_key`), `torchvision_dataset`"
        )
