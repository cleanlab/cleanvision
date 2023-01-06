import numpy as np
import pytest

from cleanvision.issue_managers import IssueType
from cleanvision.issue_managers.image_property_helpers import (
    BrightnessHelper,
    calculate_brightness,
)


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


class TestBrightnessHelper:
    @pytest.fixture
    def helper(self):
        return BrightnessHelper(IssueType.LIGHT)

    def test_init(self, helper):
        assert isinstance(helper, BrightnessHelper)
        assert hasattr(helper, "issue_type")

    @pytest.mark.parametrize(
        "mock_mean,expected_output",
        [
            [[100], 0.39216],
            [[100, 200, 50], 0.68172],
        ],
        ids=["gray", "rgb"],
    )
    def test_calculate(self, helper, monkeypatch, mock_mean, expected_output):
        from PIL import ImageStat

        class MockStat:
            def __init__(self, *args, **kwargs):
                pass

            @property
            def mean(self):
                return mock_mean

        monkeypatch.setattr(ImageStat, "Stat", MockStat)

        cur_bright = helper.calculate("my_image")
        assert cur_bright == pytest.approx(expected=expected_output, abs=1e-5)

    def test_normalize(self, helper, monkeypatch):
        raw_scores = [0.5, 0.3, 1.0, 1.2, 0.9, 0.1, 0.2]
        expected_output = np.array([0.5, 0.3, 1.0, 1.0, 0.9, 0.1, 0.2])

        with monkeypatch.context() as m:
            m.setattr(helper, "issue_type", IssueType.DARK)
            normalized_scores = helper.normalize(raw_scores)
            assert all(normalized_scores == expected_output)

        normalized_scores = helper.normalize(raw_scores)
        assert all(normalized_scores == 1 - expected_output)

    @pytest.mark.parametrize(
        "scores,threshold,expected_mark",
        [
            [0.23, 0.4, [True]],
            [0.6, 0.5, [False]],
            [
                np.array([0.1, 0.2, 0.3, 0.4]),
                0.3,
                np.array([True, True, False, False]),
            ],
        ],
    )
    def test_mark_issue(self, helper, scores, threshold, expected_mark):
        mark = helper.mark_issue(np.array(scores), threshold)
        assert all(mark == expected_mark)
