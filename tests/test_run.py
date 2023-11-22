import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cleanvision import Imagelab
from cleanvision.dataset.fsspec_dataset import FSDataset
from cleanvision.issue_managers.image_property import BrightnessProperty
from cleanvision.issue_managers.image_property_issue_manager import (
    compute_scores_wrapper,
)


@pytest.fixture(scope="module")
def imagelab():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./data")
    imagelab = Imagelab(data_path=data_path)
    return imagelab


@pytest.mark.usefixtures("set_plt_show")
def test_imagelab_run(capsys, imagelab):
    imagelab.find_issues()
    imagelab.report()
    captured = capsys.readouterr()

    assert len(imagelab.issue_summary) == 9
    assert len(imagelab.issues) == 10
    assert "statistics" in imagelab.info
    for key in [
        "color_space",
        "entropy",
        "brightness",
        "blurriness",
        "aspect_ratio",
        "size",
    ]:
        assert key in imagelab.info["statistics"]

    assert imagelab.info["exact_duplicates"]["num_sets"] == 1
    assert imagelab.info["near_duplicates"]["num_sets"] == 1
    assert len(imagelab.issues.query("is_dark_issue")) == 1
    assert len(imagelab.issues.query("is_odd_size_issue")) == 1
    assert len(imagelab.issues.query("is_odd_aspect_ratio_issue")) == 1
    assert len(imagelab.issues.query("is_low_information_issue")) == 2
    assert len(imagelab.issues.query("is_light_issue")) == 1
    assert len(imagelab.issues.query("is_grayscale_issue")) == 1
    assert len(imagelab.issues.query("is_blurry_issue")) == 1
    assert len(imagelab.issues.query("is_exact_duplicates_issue")) == 2
    assert len(imagelab.issues.query("is_near_duplicates_issue")) == 2
    assert "Issues found in images in order of severity in the dataset" in captured.out
    assert "Number of examples with this issue:" in captured.out
    assert "Examples representing most severe instances of this issue:" in captured.out


def test_get_stats(imagelab):
    stats = imagelab.get_stats()
    assert stats == imagelab.info["statistics"]


def test_incremental_issue_finding(generate_local_dataset, len_dataset):
    imagelab = Imagelab(data_path=generate_local_dataset)
    issue_types = {"near_duplicates": {}}
    imagelab.find_issues(issue_types)

    assert len(imagelab.issue_summary) == 1
    assert "near_duplicates" in imagelab.issue_summary["issue_type"].values
    assert len(imagelab.issues) == len_dataset
    assert set(["is_near_duplicates_issue", "near_duplicates_score"]) == set(
        imagelab.issues.columns
    )
    assert set(["statistics", "near_duplicates", "exact_duplicates"]) == set(
        imagelab.info.keys()
    )

    issue_types = {"light": {}, "low_information": {}}
    imagelab.find_issues(issue_types)

    assert len(imagelab.issue_summary) == 3
    assert set(["light", "low_information", "near_duplicates"]) == set(
        imagelab.issue_summary["issue_type"].values
    )
    assert len(imagelab.issues) == len_dataset
    assert set(
        [
            "is_near_duplicates_issue",
            "near_duplicates_score",
            "is_light_issue",
            "light_score",
            "is_low_information_issue",
            "low_information_score",
        ]
    ) == set(imagelab.issues.columns)
    assert set(
        [
            "statistics",
            "near_duplicates",
            "exact_duplicates",
            "light",
            "low_information",
        ]
    ) == set(imagelab.info.keys())
    assert set(["brightness", "entropy"]) == set(imagelab.info["statistics"].keys())
    assert set(
        [
            "brightness_perc_99",
            "brightness_perc_95",
            "brightness_perc_90",
            "brightness_perc_15",
            "brightness_perc_10",
            "brightness_perc_5",
            "brightness_perc_1",
        ]
    ) == set(imagelab.info["light"].keys())


@pytest.mark.parametrize(
    "n_jobs_given",
    [1, 2, None],
)
def test_jobs(generate_local_dataset, n_jobs_given):
    imagelab = Imagelab(data_path=generate_local_dataset)

    class MockPool:
        def __init__(self, n_jobs):
            if n_jobs_given:
                assert n_jobs == n_jobs_given
            else:
                assert n_jobs > 1

        def __enter__(self):
            pass

        def __exit__(self):
            pass

    imagelab.find_issues(n_jobs=n_jobs_given)


def test_compute_scores(generate_single_image_file):
    dataset = FSDataset(filepaths=[generate_single_image_file])
    image_properties = {
        "dark": BrightnessProperty("dark"),
        "light": BrightnessProperty("light"),
    }
    args = {
        "to_compute": ["dark", "light"],
        "index": generate_single_image_file,
        "dataset": dataset,
        "image_properties": image_properties,
    }
    _ = compute_scores_wrapper(args)


def test_hf_dataset_run(hf_dataset):
    imagelab = Imagelab(hf_dataset=hf_dataset, image_key="image")
    imagelab.find_issues()
    assert len(imagelab.issues.columns) == 18
    assert len(imagelab.issues) == len(hf_dataset)


def test_torch_dataset_run(torch_dataset):
    imagelab = Imagelab(torchvision_dataset=torch_dataset)
    imagelab.find_issues()
    assert len(imagelab.issues.columns) == 18
    assert len(imagelab.issues) == len(torch_dataset)


def test_run_imagelab_given_filepaths(generate_local_dataset, images_per_class):
    files = os.listdir(generate_local_dataset / "class_0")
    filepaths = [os.path.join(generate_local_dataset / "class_0", f) for f in files]
    imagelab = Imagelab(filepaths=filepaths)
    imagelab.find_issues()
    assert len(imagelab.issues.columns) == 18
    assert len(imagelab.issues) == images_per_class


def test_raise_error_given_duplicate_filepaths():
    filepaths = ["/home/user/file0.jpg", "/home/user/file1.jpg", "/home/user/file0.jpg"]
    with pytest.raises(ValueError, match="Duplicate"):
        Imagelab(filepaths=filepaths)


def test_s3_dataset(capsys, get_example_s3_filepaths):
    imagelab = Imagelab(filepaths=get_example_s3_filepaths, storage_opts={"anon": True})
    imagelab.find_issues(
        issue_types={
            "near_duplicates": {},
        },
        n_jobs=1,
    )
    assert len(imagelab.issues.columns) == 2
    captured = capsys.readouterr()
    imagelab.report()
    captured = capsys.readouterr()
    assert len(captured) > 0


def test_filepath_dataset_size_negative(generate_local_dataset_once, images_per_class):
    """

    All images are same size, so no image should have an size issue

    """
    files = os.listdir(generate_local_dataset_once / "class_0")
    filepaths = [
        os.path.join(generate_local_dataset_once / "class_0", f) for f in files
    ]
    imagelab = Imagelab(filepaths=filepaths)
    imagelab.find_issues()
    assert len(imagelab.issues.columns) == 18
    assert len(imagelab.issues[imagelab.issues["is_odd_size_issue"]]) == 0


def test_odd_size_too_large_image(generate_local_dataset_once):
    """
    Size issue is defined based on the area of an image. If the sqrt(width * height) is larger than the median
    sqrt(width * height)*threshold(default 10),is_odd_size_issue is set to True. In this example, the median area is sqrt(300x300) so 300.
    An image with 3001 x 3001 has an value of 3001 so its more than 10x smaller and thus should be flagged.
    """
    arr = np.random.randint(low=0, high=256, size=(3001, 3001, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    img.save(Path(generate_local_dataset_once / "class_0" / "larger.png"))

    files = os.listdir(generate_local_dataset_once / "class_0")
    filepaths = [
        os.path.join(generate_local_dataset_once / "class_0", f) for f in files
    ]
    imagelab = Imagelab(filepaths=filepaths)
    imagelab.find_issues()
    assert len(imagelab.issues.columns) == 18
    assert len(imagelab.issues[imagelab.issues["is_odd_size_issue"]]) == 1


@pytest.mark.usefixtures("set_plt_show")
def test_odd_size_too_small_image(generate_local_dataset_once):
    """
    Size issue is defined based on the area of an image. If the sqrt(width * height) is larger than the median
    sqrt(width * height)*threshold(default 10),is_odd_size_issue is set to True. In this example, the median area is sqrt(300x300) so 300.
    An image with 29 x 29 has an value of 29 so its more than 10x smaller and thus should be flagged.
    """
    arr = np.random.randint(
        low=0,
        high=256,
        size=(29, 29, 3),
        dtype=np.uint8,  # 30 x 30 pixel image should be detected
    )
    img = Image.fromarray(arr, mode="RGB")
    img.save(Path(generate_local_dataset_once / "class_0" / "smaller.png"))

    files = os.listdir(generate_local_dataset_once / "class_0")
    filepaths = [
        os.path.join(generate_local_dataset_once / "class_0", f) for f in files
    ]
    imagelab = Imagelab(filepaths=filepaths)
    imagelab.find_issues()
    assert len(imagelab.issues.columns) == 18
    assert len(imagelab.issues[imagelab.issues["is_odd_size_issue"]]) == 1


def test_custom_threshold_for_odd_size(generate_local_dataset_once):
    """
    With default threshold the small image would be flagged (See test_filepath_dataset_size_to_small). However,
     with a custom threshold of 11 instead of 10, the imaage is within the allowed range and should not be flagged.
    """
    arr = np.random.randint(
        low=0,
        high=256,
        size=(29, 29, 3),
        dtype=np.uint8,  # 29 x 29 pixel image should not be detected with threshold 11
    )
    img = Image.fromarray(arr, mode="RGB")
    img.save(Path(generate_local_dataset_once / "class_0" / "smaller.png"))

    files = os.listdir(generate_local_dataset_once / "class_0")
    filepaths = [
        os.path.join(generate_local_dataset_once / "class_0", f) for f in files
    ]
    imagelab = Imagelab(filepaths=filepaths)
    imagelab.find_issues({"odd_size": {"threshold": 11.0}})
    assert len(imagelab.issues.columns) == 2  # Only size
    assert len(imagelab.issues[imagelab.issues["is_odd_size_issue"]]) == 0


def test_list_default_issue_types():
    default_issues = Imagelab.list_default_issue_types()
    assert set(default_issues) == set(
        [
            "dark",
            "light",
            "odd_aspect_ratio",
            "low_information",
            "exact_duplicates",
            "near_duplicates",
            "blurry",
            "grayscale",
            "odd_size",
        ]
    )


def test_new_issue_registration(generate_local_dataset, len_dataset):
    from docs.source.tutorials.custom_issue_manager import CustomIssueManager

    issue_name = CustomIssueManager.issue_name
    possible_issues = Imagelab.list_possible_issue_types()
    assert len(possible_issues) == 10
    assert issue_name in possible_issues

    imagelab = Imagelab(data_path=generate_local_dataset)
    issue_types = {issue_name: {}}
    imagelab.find_issues(issue_types)

    assert set(imagelab.issue_summary["issue_type"].values) == set([issue_name])
    assert len(imagelab.issues) == len_dataset
    assert set(["is_custom_issue", "custom_score"]) == set(imagelab.issues.columns)
    assert set(["statistics", "custom"]) == set(imagelab.info.keys())
