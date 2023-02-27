import pytest
import os
import argparse
from PIL import Image
import numpy as np

from cleanvision.imagelab import Imagelab
from examples.custom_issue_manager import CustomIssueManager

# main_script_dir = os.path.dirname(__file__)
# rel_path = "resources/images/phone.png"
# PHONE_IMAGE_PATH = os.path.join(main_script_dir, rel_path)
#
# dataset_path = "../"

# def test_something_else(tmp_path):
#     #create a file "myfile" in "mydir" in temp directory
#     f1 = tmp_path / "mydir/myfile"
#     f1.parent.mkdir() #create a directory "mydir" in temp folder (which is the parent directory of "myfile"
#     f1.touch() #create a file "myfile" in "mydir"
#
#
#     #write to file as normal
#     f1.write_text("text to myfile")
#
#     assert f1.read_text() == "text to myfile"

# content of test_tmp_path.py
CONTENT = "content"


def generate_image(arr=None):
    if arr is None:
        arr = np.random.randint(
            low=0,
            high=256,
            size=(300, 300, 3),
            dtype=np.uint8
        )
    img = Image.fromarray(arr, mode='RGB')
    return img


@pytest.fixture(scope="session")
def generate_single_image_file(tmpdir_factory, img_name="img.png", arr=None):
    """Generates a single temporary image for testing"""
    img = generate_image(arr)
    fn = tmpdir_factory.mktemp("data").join(img_name)
    img.save(str(fn))
    return str(fn)


@pytest.fixture(scope="session")
def generate_n_image_files(tmpdir_factory, n=40):
    """Generates n temporary images for testing and returns dir of images"""
    filename_list = []
    tmp_image_dir = tmpdir_factory.mktemp("data")
    for i in range(n):
        img = generate_image()
        img_name = f"{i}.png"
        fn = tmp_image_dir.join(img_name)
        img.save(str(fn))
        filename_list.append(str(fn))
    return str(tmp_image_dir)


# contents of test_image.py
def test_histogram(generate_single_image_file):
    img = Image.open(generate_single_image_file)
    # compute and test histogram


@pytest.mark.usefixtures('generate_n_image_files')
def test_example1(capsys, generate_n_image_files):
    imagelab = Imagelab(data_path=generate_n_image_files)  # initialize imagelab
    imagelab.list_default_issue_types()  # list default checks

    ## ==== Test list_default_issue_types() lists all default isssue types====
    DEFAULT_ISSUE_TYPES = ['dark', 'light', 'odd_aspect_ratio', 'low_information', 'exact_duplicates',
                           'near_duplicates', 'blurry', 'grayscale']
    captured = capsys.readouterr()

    for issue_type in DEFAULT_ISSUE_TYPES:
        assert issue_type in captured.out

    for issue_type in imagelab._issue_types:
        assert issue_type in captured.out

    ## ===[TODO] Test visualize produces some result ===
    # imagelab.visualize()  # visualize random images in dataset

    ## === Test find_issues() finds issues in dataset ===
    assert len(imagelab.issue_summary) == 0
    imagelab.find_issues()  # Find issues in the dataset
    assert len(imagelab.issue_summary) > 0

    ## === Test report() produces print output ===
    captured = capsys.readouterr()
    imagelab.report()
    captured = capsys.readouterr()
    assert len(captured) > 0

    ## === [TODO] Test visualize works as needed ===
    # imagelab.visualize(
    #     issue_types=["blurry"], num_images=8
    # )  # visualize images that have specific issues
    #
    # # Get all images with blurry issue type
    # blurry_images = imagelab.issues[
    #     imagelab.issues["is_blurry_issue"] == True
    #     ].index.to_list()
    # imagelab.visualize(image_files=blurry_images)  # visualize the given image files

    ## === Test miscellaneous extra information about datasset
    assert (list(imagelab.info.keys()) == ['statistics', 'exact_duplicates', 'near_duplicates'])
    for key in ['color_space', 'entropy', 'brightness', 'blurriness', 'aspect_ratio']:
        assert key in list(imagelab.info["statistics"].keys())

@pytest.mark.usefixtures('generate_n_image_files')
def test_example2(capsys, generate_n_image_files):
    imagelab = Imagelab(data_path=generate_n_image_files)  # initialize imagelab
    issue_types = {"near_duplicates": {}}
    imagelab.find_issues(issue_types)
    imagelab.report()
    save_folder = generate_n_image_files + '/T_save_folder/'

    imagelab.save(
        save_folder
    )  # optional, just included to show how to save/load this as a file

    # Check for additional types of issues using existing Imagelab
    imagelab = Imagelab.load(save_folder, generate_n_image_files)
    issue_types = {"light": {}, "low_information": {}}
    imagelab.find_issues(issue_types)
    imagelab.report()

    # Check for an issue with a different threshold
    issue_types = {"dark": {"threshold": 0.2}}
    imagelab.find_issues(issue_types)
    imagelab.report(issue_types=issue_types.keys())  # report only specific issues

@pytest.mark.usefixtures('generate_n_image_files')
def test_example3(capsys, generate_n_image_files):
    imagelab = Imagelab(data_path=generate_n_image_files)
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


@pytest.mark.usefixtures('generate_n_image_files')
def test_example4(capsys, generate_n_image_files):
    imagelab = Imagelab(data_path=generate_n_image_files)
    issue_name = CustomIssueManager.issue_name
    imagelab.list_possible_issue_types()

    issue_types = {issue_name: {}}
    imagelab.find_issues(issue_types)  # check for custom issue type

    imagelab.find_issues()  # also check for default issue types
    imagelab.report()
