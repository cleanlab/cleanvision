import random
from unittest.mock import patch

import pytest
from PIL import Image

import cleanvision
from cleanvision import Imagelab
from cleanvision.utils.utils import get_filepaths


@pytest.fixture()
def folder_imagelab(generate_local_dataset):
    imagelab = Imagelab(data_path=generate_local_dataset)
    return imagelab


@pytest.fixture()
def hf_imagelab(hf_dataset):
    imagelab = Imagelab(hf_dataset=hf_dataset, image_key="image")
    return imagelab


@pytest.fixture()
def torch_imagelab(torch_dataset):
    imagelab = Imagelab(torchvision_dataset=torch_dataset)
    return imagelab


def test_visualize_sample_images(
    generate_local_dataset, monkeypatch, capsys, folder_imagelab
):
    filepaths = get_filepaths(generate_local_dataset)[:3]

    def mock_sample(*args, **kwargs):
        return filepaths

    monkeypatch.setattr(random, "sample", mock_sample)

    with patch.object(
        cleanvision.utils.viz_manager.VizManager, "individual_images", return_value=None
    ) as mock_viz_method:
        folder_imagelab.visualize()

    captured = capsys.readouterr()
    assert "Sample images from the dataset" in captured.out

    images = [Image.open(filepath) for filepath in filepaths]
    mock_viz_method.call_args.args[0] == images


def test_visualize_sample_images_hf_dataset(
    monkeypatch, capsys, hf_imagelab, hf_dataset
):
    sample_indices = [0, 1, 2]

    def mock_sample(*args, **kwargs):
        return sample_indices

    monkeypatch.setattr(random, "sample", mock_sample)

    with patch.object(
        cleanvision.utils.viz_manager.VizManager, "individual_images", return_value=None
    ) as mock_viz_method:
        hf_imagelab.visualize()

    captured = capsys.readouterr()
    assert "Sample images from the dataset" in captured.out

    images = [hf_dataset[i]["image"] for i in sample_indices]
    mock_viz_method.call_args.args[0] == images


def test_visualize_sample_images_torch_dataset(
    monkeypatch, capsys, torch_imagelab, torch_dataset
):
    sample_indices = [0, 1, 2]

    def mock_sample(*args, **kwargs):
        return sample_indices

    monkeypatch.setattr(random, "sample", mock_sample)

    with patch.object(
        cleanvision.utils.viz_manager.VizManager, "individual_images", return_value=None
    ) as mock_viz_method:
        torch_imagelab.visualize()

    captured = capsys.readouterr()
    assert "Sample images from the dataset" in captured.out

    images = [torch_dataset[i][0] for i in sample_indices]
    mock_viz_method.call_args.args[0] == images


def test_visualize_indices(folder_imagelab, generate_local_dataset):
    filepaths = get_filepaths(generate_local_dataset)[:3]

    with patch.object(
        cleanvision.utils.viz_manager.VizManager, "individual_images", return_value=None
    ) as mock_viz_method:
        folder_imagelab.visualize(indices=filepaths)
        folder_imagelab.visualize(image_files=filepaths)

    images = [Image.open(filepath) for filepath in filepaths]
    mock_viz_method.call_args.args[0] == images
    mock_viz_method.call_count == 2


def test_visualize_indices_hf(hf_imagelab, hf_dataset):
    sample_indices = [0, 1, 2]

    with patch.object(
        cleanvision.utils.viz_manager.VizManager, "individual_images", return_value=None
    ) as mock_viz_method:
        hf_imagelab.visualize(indices=sample_indices)

    images = [hf_dataset[i]["image"] for i in sample_indices]
    mock_viz_method.call_args.args[0] == images


def test_visualize_indices_torch(torch_imagelab, torch_dataset):
    sample_indices = [0, 1, 2]

    with patch.object(
        cleanvision.utils.viz_manager.VizManager, "individual_images", return_value=None
    ) as mock_viz_method:
        torch_imagelab.visualize(indices=sample_indices)

    images = [torch_dataset[i][0] for i in sample_indices]
    mock_viz_method.call_args.args[0] == images
