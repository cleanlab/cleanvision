import os

import torchvision
from datasets import load_dataset

from cleanvision.dataset.folder_dataset import FolderDataset
from cleanvision.dataset.hf_dataset import HFDataset
from cleanvision.dataset.torch_dataset import TorchDataset
from cleanvision.dataset.utils import build_dataset


class TestDataset:
    def test_build_folder_dataset(
        self, generate_local_dataset, n_classes, images_per_class
    ):
        dataset = build_dataset(data_path=generate_local_dataset)
        assert isinstance(dataset, FolderDataset)
        assert len(dataset.index) == n_classes * images_per_class
        assert isinstance(dataset.index[0], str)

    def test_build_filepaths_dataset(self, generate_local_dataset, images_per_class):
        files = os.listdir(generate_local_dataset / "class_0")
        filepaths = [os.path.join(generate_local_dataset / "class_0", f) for f in files]
        dataset = build_dataset(filepaths=filepaths)
        assert isinstance(dataset, FolderDataset)
        assert len(dataset.index) == images_per_class
        assert isinstance(dataset.index[0], str)

    def test_build_hf_dataset(
        self, generate_local_dataset, n_classes, images_per_class
    ):
        hf_dataset = load_dataset(
            "imagefolder", data_dir=generate_local_dataset, split="train"
        )
        dataset = build_dataset(hf_dataset=hf_dataset, image_key="image")
        assert isinstance(dataset, HFDataset)
        assert len(dataset.index) == n_classes * images_per_class
        assert isinstance(dataset.index[0], int)

    def test_build_torch_dataset(
        self, generate_local_dataset, n_classes, images_per_class
    ):
        torch_ds = torchvision.datasets.ImageFolder(root=generate_local_dataset)
        dataset = build_dataset(torchvision_dataset=torch_ds)
        assert isinstance(dataset, TorchDataset)
        assert len(dataset.index) == n_classes * images_per_class
        assert isinstance(dataset.index[0], int)
