from unittest.mock import patch
from cleanvision import Imagelab
from cleanvision.utils.utils import get_filepaths
import random
from PIL import Image
from datasets import load_dataset
import cleanvision
import torchvision


def test_visualize_sample_images(generate_local_dataset, monkeypatch, capsys):
    imagelab = Imagelab(data_path=generate_local_dataset)
    filepaths = get_filepaths(generate_local_dataset)[:3]

    def mock_sample(*args, **kwargs):
        return filepaths

    monkeypatch.setattr(random, "sample", mock_sample)

    with patch.object(
        cleanvision.utils.viz_manager.VizManager, "individual_images", return_value=None
    ) as mock_viz_method:
        imagelab.visualize()

    captured = capsys.readouterr()
    assert "Sample images from the dataset" in captured.out

    images = [Image.open(filepath) for filepath in filepaths]
    mock_viz_method.call_args.args[0] == images


def test_visualize_sample_images_hf_dataset(
    generate_local_dataset, monkeypatch, capsys
):
    hf_dataset = load_dataset(
        "imagefolder", data_dir=generate_local_dataset, split="train"
    )
    imagelab = Imagelab(hf_dataset=hf_dataset, image_key="image")
    sample_indices = [0, 1, 2]

    def mock_sample(*args, **kwargs):
        return sample_indices

    monkeypatch.setattr(random, "sample", mock_sample)

    with patch.object(
        cleanvision.utils.viz_manager.VizManager, "individual_images", return_value=None
    ) as mock_viz_method:
        imagelab.visualize()

    captured = capsys.readouterr()
    assert "Sample images from the dataset" in captured.out

    images = [hf_dataset[i]["image"] for i in sample_indices]
    mock_viz_method.call_args.args[0] == images


def test_visualize_sample_images_torch_dataset(
    generate_local_dataset, monkeypatch, capsys
):
    torch_ds = torchvision.datasets.ImageFolder(root=generate_local_dataset)
    imagelab = Imagelab(torchvision_dataset=torch_ds)
    sample_indices = [0, 1, 2]

    def mock_sample(*args, **kwargs):
        return sample_indices

    monkeypatch.setattr(random, "sample", mock_sample)

    with patch.object(
        cleanvision.utils.viz_manager.VizManager, "individual_images", return_value=None
    ) as mock_viz_method:
        imagelab.visualize()

    captured = capsys.readouterr()
    assert "Sample images from the dataset" in captured.out

    images = [torch_ds[i][0] for i in sample_indices]
    mock_viz_method.call_args.args[0] == images
