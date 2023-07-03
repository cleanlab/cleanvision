from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from cleanvision.dataset.base_dataset import Dataset
from cleanvision.dataset.fsspec_dataset import FSDataset
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
    storage_opts: Optional[Dict[str, str]] = {},
) -> Dataset:
    if data_path:
        return FSDataset(data_folder=data_path, storage_opts=storage_opts)
    elif filepaths:
        return FSDataset(filepaths=filepaths, storage_opts=storage_opts)
    elif hf_dataset and image_key:
        return HFDataset(hf_dataset, image_key)
    elif torchvision_dataset:
        return TorchDataset(torchvision_dataset)
    else:
        raise ValueError(
            "You must specify some argument among the following: `data_path`, `filepaths`, (`hf_dataset`, `image_key`), `torchvision_dataset`"
        )
