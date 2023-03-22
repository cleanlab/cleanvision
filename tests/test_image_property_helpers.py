import numpy as np
import pandas as pd
import pytest
from PIL import Image

import cleanvision
from cleanvision.issue_managers import IssueType
from cleanvision.issue_managers.image_property import (
    BrightnessProperty,
    calculate_brightness,
    get_image_mode,
    calc_aspect_ratio,
    calc_entropy,
    calc_blurriness,
    get_edges,
)
from cleanvision.utils.utils import get_is_issue_colname, get_score_colname


@pytest.mark.parametrize(
    "rgb,expected_brightness",
    [
        [(0, 0, 0), 0],
        [(255, 255, 255), 1],
        [(255, 0, 0), 0.49092],
        [(0, 255, 0), 0.83126],
        [(0, 0, 255), 0.26077],
    ],
    ids=["min", "max", "red", "green", "blue"],
)
def test_calculate_brightness(rgb, expected_brightness):
    brightness = calculate_brightness(*rgb)
    assert brightness == pytest.approx(expected=expected_brightness, abs=1e-5)


def test_calc_aspect_ratio():
    img = Image.new("RGB", (200, 200), (255, 0, 0))
    size_score = calc_aspect_ratio(img)  # min(width/height,height/width)
    assert size_score == 1


def test_calc_entropy():
    img = Image.new("RGB", (200, 200), (255, 0, 0))
    entropy_score = calc_entropy(img)  # min(width/height,height/width)
    assert entropy_score == img.entropy()


def test_calc_bluriness():
    img = Image.new("RGB", (200, 200), (255, 0, 0))
    edges = get_edges(img)
    blurriness = calc_blurriness(img, 512)
    assert isinstance(edges, Image.Image)
    assert isinstance(blurriness, float)


@pytest.mark.parametrize(
    "image,expected_mode",
    [
        [Image.new("RGB", (164, 164), (255, 255, 255)), "RGB"],
        [Image.new("RGB", (164, 164)), "RGB"],
        [Image.new("L", (164, 164)), "L"],
        [Image.new("RGB", (164, 164), (255, 160, 255)), "RGB"],
    ],
    ids=["white", "black", "grayscale", "rgb"],
)
def test_get_image_mode(image, expected_mode):
    mode = get_image_mode(image)
    assert mode == expected_mode


class TestBrightnessHelper:
    @pytest.fixture
    def image_property(self):
        return BrightnessProperty(IssueType.LIGHT.value)

    def test_init(self, image_property):
        assert isinstance(image_property, BrightnessProperty)
        assert hasattr(image_property, "issue_type")

    def test_calculate(self, image_property, monkeypatch):
        image = Image.new("RGB", (2, 3))

        def mock_perc_brightness(image, percentiles):
            return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        def mock_avg_brightness(image):
            return 0.1

        monkeypatch.setattr(
            cleanvision.issue_managers.image_property,
            "calc_percentile_brightness",
            mock_perc_brightness,
        )
        monkeypatch.setattr(
            cleanvision.issue_managers.image_property,
            "calc_avg_brightness",
            mock_avg_brightness,
        )

        raw_values = image_property.calculate(image)
        assert raw_values["brightness_perc_15"] == 0.4
        assert raw_values["brightness"] == 0.1

    @pytest.mark.parametrize(
        "issue_type, expected_output",
        [("light", [0.5, 0.7, 0.1, 0.9, 0.8]), ("dark", [0.5, 0.3, 0.9, 0.1, 0.2])],
    )
    def test_get_scores(self, image_property, issue_type, expected_output):
        raw_values = [0.5, 0.3, 0.9, 0.1, 0.2]
        raw_scores = pd.DataFrame(
            {"brightness_perc_5": raw_values, "brightness_perc_99": raw_values}
        )
        expected_scores = pd.DataFrame({get_score_colname(issue_type): expected_output})

        scores = image_property.get_scores(raw_scores, issue_type)
        pd.testing.assert_frame_equal(scores, expected_scores)

    @pytest.mark.parametrize(
        "scores,threshold,expected_mark",
        [
            [
                pd.DataFrame(
                    data={get_score_colname("fake_issue"): [0.1, 0.2, 0.3, 0.4]}
                ),
                0.3,
                pd.DataFrame(
                    data={
                        get_is_issue_colname("fake_issue"): [True, True, False, False]
                    }
                ),
            ],
        ],
    )
    def test_mark_issue(self, image_property, scores, threshold, expected_mark):
        mark = image_property.mark_issue(scores, threshold, "fake_issue")
        assert all(mark == expected_mark)
