import pytest
from PIL import Image
import numpy as np
from pathlib import Path

from cleanvision.issue_managers import IssueType
from cleanvision.dataset.folder_dataset import FolderDataset
from cleanvision.issue_managers.size_issue_manager import (
    SizeIssueManager,
    get_size,
)

SIZE = IssueType.SIZE.value


class TestSizeIssueManager:
    @pytest.fixture
    def issue_manager(self):
        return SizeIssueManager()

    @pytest.mark.parametrize(
        "params,expected_params",
        [({}, {"size": {"ratio": 5.0}}), ({"ratio": 6.3}, {"size": {"ratio": 6.3}})],
        ids=["use default", "set params"],
    )
    def test_set_params(self, params, expected_params, issue_manager):
        """Tests DuplicateIssueManager.set_params(). Cases covered:
        1. If no params are specified for an issue_type, default params are used
        2. If params are specified, those specific params are updated, for the remaining ones default values are used
        """
        issue_manager.update_params(params)
        assert issue_manager.params == expected_params

    @pytest.mark.parametrize(
        "ratio,additional_image_size",
        [
            (
                5.0,
                (1700, 1700, 3),  # h,w
            ),
            (
                5.0,
                (1500, 1500, 3),  # h,w
            ),
            (
                3.0,
                (1500, 1500, 3),  # h,w
            ),
            (
                5.0,
                (2000, 300, 3),  # h,w
            ),
            (
                5.0,
                (300, 2000, 3),  # h,w
            ),
        ],
        ids=[
            "ratio 5 with width and height over 5",
            "ratio 5 with width and height below 5",
            "ratio 3 with width and height over 3",
            "ratio 5 and 1 image with width over 5",
            "ratio 5 and 1 image with height over 5",
        ],
    )
    def test_issues_larger(
        self, ratio, additional_image_size, generate_local_dataset_once, issue_manager
    ):
        # Add one with the size specified in the params to the 40 * 300x300 images
        arr = np.random.randint(
            low=0, high=256, size=additional_image_size, dtype=np.uint8
        )  # Add one image with 10x width and height
        img = Image.fromarray(arr, mode="RGB")
        img.save(Path(generate_local_dataset_once, "larger.png"))
        dataset = FolderDataset(generate_local_dataset_once)

        # Calculate avg. height/width of all the images
        avg_height = (40 * 300 + additional_image_size[0]) / 41.0
        avg_width = (40 * 300 + additional_image_size[1]) / 41.0

        # Check if with the specified ratio, the large one is out of bounds
        height_issue_count = 1 if additional_image_size[0] > ratio * avg_height else 0
        width_issue_count = 1 if additional_image_size[1] > ratio * avg_width else 0

        issue_manager.find_issues(
            dataset=dataset, params={"ratio": ratio}, imagelab_info={}, n_jobs=1
        )
        summary = issue_manager.summary

        # Assert num images in summary is right
        assert (
            summary[summary["issue_type"] == "width"]["num_images"].values[0]
            == width_issue_count
        )
        assert (
            summary[summary["issue_type"] == "height"]["num_images"].values[0]
            == height_issue_count
        )

        issues = issue_manager.issues

        # Assert is_width_issue and is_height_issue has the right count
        assert len(issues[issues["is_height_issue"] == True]) == height_issue_count
        assert len(issues[issues["is_width_issue"] == True]) == width_issue_count

        # Assert the right number of images is over the ratio*avg in the issues
        # The height_score_raw is (img_height / avg_height of all images)
        # The width_score_raw is (img_width / avg_width of all images)
        assert len(issues[issues["height_score_raw"] > ratio]) == height_issue_count
        assert len(issues[issues["width_score_raw"] > ratio]) == width_issue_count


def test_get_size(generate_local_dataset_once):
    dataset = FolderDataset(generate_local_dataset_once)
    for idx in dataset.index:
        assert get_size(idx, dataset) == {"index": idx, "height": 300, "width": 300}
