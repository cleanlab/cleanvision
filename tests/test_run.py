import os

import pytest
import torchvision
from datasets import load_dataset

from cleanvision.dataset.folder_dataset import FolderDataset
from cleanvision.imagelab import Imagelab
from cleanvision.issue_managers.image_property import BrightnessProperty
from cleanvision.issue_managers.image_property_issue_manager import (
    compute_scores_wrapper,
)
from docs.source.tutorials.custom_issue_manager import CustomIssueManager


@pytest.mark.usefixtures("set_plt_show")
def test_example1(capsys, generate_local_dataset):
    imagelab = Imagelab(data_path=generate_local_dataset)  # initialize imagelab
    imagelab.list_default_issue_types()  # list default checks

    # ==== Test list_default_issue_types() lists all default isssue types====
    DEFAULT_ISSUE_TYPES = [
        "dark",
        "light",
        "odd_aspect_ratio",
        "low_information",
        "exact_duplicates",
        "near_duplicates",
        "blurry",
        "grayscale",
    ]
    captured = capsys.readouterr()

    for issue_type in DEFAULT_ISSUE_TYPES:
        assert issue_type in captured.out

    for issue_type in imagelab._issue_types:
        assert issue_type in captured.out

    # ===[TODO] Test visualize produces some result ===
    # imagelab.visualize()  # visualize random images in dataset

    # === Test find_issues() finds issues in dataset ===
    assert len(imagelab.issue_summary) == 0
    imagelab.find_issues()  # Find issues in the dataset
    assert len(imagelab.issue_summary) > 0

    # === Test report() produces print output ===
    captured = capsys.readouterr()
    imagelab.report()
    captured = capsys.readouterr()
    assert len(captured) > 0

    # === [TODO] Test visualize works as needed ===
    # imagelab.visualize(
    #     issue_types=["blurry"], num_images=8
    # )  # visualize images that have specific issues
    #
    # # Get all images with blurry issue type
    # blurry_images = imagelab.issues[
    #     imagelab.issues["is_blurry_issue"] == True
    #     ].index.to_list()
    # imagelab.visualize(image_files=blurry_images)  # visualize the given image files

    # === Test miscellaneous extra information about datasset
    assert set(imagelab.info.keys()) == set(DEFAULT_ISSUE_TYPES + ["statistics"])
    for key in ["color_space", "entropy", "brightness", "blurriness", "aspect_ratio"]:
        assert key in list(imagelab.info["statistics"].keys())


@pytest.mark.usefixtures("set_plt_show")
def test_example2(generate_local_dataset, tmp_path):
    imagelab = Imagelab(data_path=generate_local_dataset)  # initialize imagelab
    issue_types = {"near_duplicates": {}}
    imagelab.find_issues(issue_types)
    imagelab.report()
    save_folder = tmp_path / "T_save_folder/"

    imagelab.save(
        save_folder
    )  # optional, just included to show how to save/load this as a file

    # Check for additional types of issues using existing Imagelab
    imagelab = Imagelab.load(save_folder, generate_local_dataset)
    issue_types = {"light": {}, "low_information": {}}
    imagelab.find_issues(issue_types)
    imagelab.report()

    # Check for an issue with a different threshold
    issue_types = {"dark": {"threshold": 0.2}}
    imagelab.find_issues(issue_types)
    imagelab.report(issue_types=issue_types.keys())  # report only specific issues


@pytest.mark.usefixtures("set_plt_show")
def test_example3(generate_local_dataset):
    imagelab = Imagelab(data_path=generate_local_dataset)
    imagelab.find_issues()
    imagelab.report(["near_duplicates"])

    issue_types = {"near_duplicates": {"hash_type": "phash"}}
    imagelab.find_issues(issue_types)
    imagelab.report(issue_types=issue_types.keys())

    # Customize report and visualize

    # Change verbosity
    imagelab.report(verbosity=3)

    # Report arg values here will overwrite verbosity defaults
    # Find top examples suffering from issues that are not present in more than 1% of the dataset
    imagelab.report(max_prevalence=0.01)

    # Increase cell_size in the grid
    imagelab.visualize(issue_types=["light"], num_images=8, cell_size=(3, 3))


@pytest.mark.usefixtures("set_plt_show")
def test_example4(generate_local_dataset):
    imagelab = Imagelab(data_path=generate_local_dataset)
    issue_name = CustomIssueManager.issue_name
    imagelab.list_possible_issue_types()

    issue_types = {issue_name: {}}
    imagelab.find_issues(issue_types)  # check for custom issue type

    imagelab.find_issues()  # also check for default issue types
    imagelab.report()


@pytest.mark.usefixtures("set_plt_show")
def test_jobs(generate_local_dataset):
    imagelab = Imagelab(data_path=generate_local_dataset)
    imagelab.find_issues(n_jobs=1)


def test_compute_scores(generate_single_image_file):
    dataset = FolderDataset(filepaths=[generate_single_image_file])
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


@pytest.mark.usefixtures("set_plt_show")
def test_example5(generate_local_dataset):
    imagelab = Imagelab(data_path=generate_local_dataset)
    imagelab.find_issues({"blurry": {}})
    imagelab.find_issues({"dark": {}})
    imagelab.report()
    # Also test the reverse direction:
    # TODO: this direction maybe can be made faster since blurry-check depends on dark-score
    imagelab = Imagelab(data_path=generate_local_dataset)
    imagelab.find_issues({"dark": {}})
    imagelab.find_issues({"blurry": {}})
    imagelab.report()


@pytest.mark.usefixtures("set_plt_show")
def test_hf_dataset_run(generate_local_dataset, n_classes, images_per_class):
    hf_dataset = load_dataset(
        "imagefolder", data_dir=generate_local_dataset, split="train"
    )
    imagelab = Imagelab(hf_dataset=hf_dataset, image_key="image")
    imagelab.find_issues()
    imagelab.report()
    assert len(imagelab.issues.columns) == 14
    assert len(imagelab.issues) == n_classes * images_per_class


@pytest.mark.usefixtures("set_plt_show")
def test_torch_dataset_run(generate_local_dataset, n_classes, images_per_class):
    torch_ds = torchvision.datasets.ImageFolder(root=generate_local_dataset)
    imagelab = Imagelab(torchvision_dataset=torch_ds)
    imagelab.find_issues()
    imagelab.report()
    assert len(imagelab.issues.columns) == 14
    assert len(imagelab.issues) == n_classes * images_per_class


@pytest.mark.usefixtures("set_plt_show")
def test_visualize_sample_images(generate_local_dataset):
    imagelab = Imagelab(data_path=generate_local_dataset)
    imagelab.visualize()


@pytest.mark.usefixtures("set_plt_show")
def test_visualize_given_imagefiles(generate_local_dataset):
    imagelab = Imagelab(data_path=generate_local_dataset)
    files = os.listdir(generate_local_dataset / "class_0")
    filepaths = [os.path.join(generate_local_dataset / "class_0", f) for f in files]
    imagelab.visualize(image_files=filepaths)


@pytest.mark.usefixtures("set_plt_show")
def test_filepath_dataset_run(generate_local_dataset, images_per_class):
    files = os.listdir(generate_local_dataset / "class_0")
    filepaths = [os.path.join(generate_local_dataset / "class_0", f) for f in files]
    imagelab = Imagelab(filepaths=filepaths)
    imagelab.find_issues()
    imagelab.report()
    assert len(imagelab.issues.columns) == 14
    assert len(imagelab.issues) == images_per_class
