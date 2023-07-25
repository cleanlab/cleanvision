import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image
from datasets import load_dataset
import torchvision


@pytest.fixture(scope="session")
def n_classes():
    return 4


@pytest.fixture(scope="session")
def images_per_class():
    return 10


@pytest.fixture(scope="session")
def len_dataset(n_classes, images_per_class):
    return n_classes * images_per_class


@pytest.fixture()
def set_plt_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)


def generate_image(arr=None):
    if arr is None:
        arr = np.random.randint(low=0, high=256, size=(300, 300, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    return img


@pytest.fixture(scope="session")
def generate_single_image_file(tmpdir_factory, img_name="img.png", arr=None):
    """Generates a single temporary image for testing"""
    img = generate_image(arr)
    fn = tmpdir_factory.mktemp("data").join(img_name)
    img.save(str(fn))
    return str(fn)


def generate_local_dataset_base(tmp_path_factory, n_classes, images_per_class):
    tmp_image_dir = tmp_path_factory.mktemp("data")
    for i in range(n_classes):
        class_dir = tmp_image_dir / f"class_{i}"
        class_dir.mkdir()
        for j in range(images_per_class):
            img = generate_image()
            img_name = f"image_{j}.png"
            fn = class_dir / img_name
            img.save(fn)
    return tmp_image_dir


@pytest.fixture()
def get_example_s3_dataset():
    """Returns a path to a small s3 dataset"""
    return "s3://cleanvision-test/test_images/"


@pytest.fixture()
def get_example_s3_filepaths():
    """Returns a list of filepaths to a small s3 dataset"""
    raw_path = "s3://amazon-berkeley-objects/images/small/aa/"
    near_dups = [
        "aa4be6a6.jpg",
        "aa6561ad.jpg",
        "aa66c206.jpg",
        "aa8a0fae.jpg",
        "aa8b5a03.jpg",
        "aabd664a.jpg",
        "aac3a821.jpg",
        "aad93004.jpg",
        "aae789f2.jpg",
        "aaea36e9.jpg",
    ]
    return [f"{raw_path}{i}" for i in near_dups]


@pytest.fixture(scope="session")
def hf_dataset(generate_local_dataset):
    hf_dataset = load_dataset(
        "imagefolder", data_dir=generate_local_dataset, split="train"
    )
    return hf_dataset


@pytest.fixture(scope="session")
def torch_dataset(generate_local_dataset):
    torch_ds = torchvision.datasets.ImageFolder(root=generate_local_dataset)
    return torch_ds


@pytest.fixture(scope="session")
def generate_local_dataset(tmp_path_factory, n_classes, images_per_class):
    """Generates n temporary images for testing and returns dir of images"""
    return generate_local_dataset_base(tmp_path_factory, n_classes, images_per_class)


@pytest.fixture(scope="function")
def generate_local_dataset_once(tmp_path_factory, n_classes, images_per_class):
    """Generates n temporary images for testing and returns dir of images"""
    return generate_local_dataset_base(tmp_path_factory, n_classes, images_per_class)
